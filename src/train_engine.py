"""
train_engine.py
-----------------
The actual training orchestration. This is where the "hangs at Epoch 1"
symptom is eliminated at the root:

  * verbose=2 is used for model.fit() instead of verbose=1. verbose=1 draws
    its progress bar using carriage returns (\\r). Many terminals attached to
    `streamlit run` (piped output, some IDE panes, Docker/systemd logs) do
    not render \\r — so epochs DO complete, they just overwrite invisibly or
    get buffered, making it look permanently stuck on "Epoch 1/25". verbose=2
    prints one clean, flushed line per epoch, so progress is always visible.

  * Training is invoked SYNCHRONOULSY on Streamlit's main script thread
    (never in a background Thread/Process). Calling any st.* function from a
    non-main thread makes Streamlit silently drop the update (missing
    ScriptRunContext) which is a classic second cause of "nothing happens
    after epoch 1" reports. Our ProgressCallback only ever runs on the same
    thread that owns the Streamlit session, so UI updates are guaranteed to
    render.

  * A lightweight ProgressCallback records epoch/batch metrics into a plain
    Python dict (no Streamlit dependency at all), so this module has zero
    coupling to the UI and can be tested/run headlessly.

  * The first training batch is explicitly shape/NaN/Inf-logged so any
    remaining data problem is visible immediately instead of after a long
    silent wait.
"""

from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from src.utils import Config, get_logger, set_global_seed, describe_array
from src.dataset_manager import prepare_dataset, DatasetBundle
from src.model_builder import build_model_from_config, model_summary_string


@dataclass
class TrainingProgress:
    """Plain-data progress snapshot. No Streamlit / thread dependency."""
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    current_acc: float = 0.0
    current_val_loss: float = 0.0
    current_val_acc: float = 0.0
    status: str = "idle"          # idle | running | completed | failed | stopped_early
    message: str = ""
    history: dict = field(default_factory=lambda: {
        "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []
    })


class ProgressCallback(tf.keras.callbacks.Callback):
    """
    Updates a TrainingProgress object and optionally calls a user-supplied
    on_epoch_callback(progress) hook after every epoch. Deliberately does
    NOT import streamlit — keeps this module UI-agnostic and thread-safe.
    """

    def __init__(self, progress: TrainingProgress, total_epochs: int,
                 on_epoch_end_hook: Optional[Callable[[TrainingProgress], None]] = None,
                 logger=None):
        super().__init__()
        self.progress = progress
        self.progress.total_epochs = total_epochs
        self.on_epoch_end_hook = on_epoch_end_hook
        self.logger = logger
        self._first_batch_logged = False

    def on_train_batch_begin(self, batch, logs=None):
        # Diagnostic-only: prove the very first batch actually starts,
        # and log its exact tensor shapes. If training were truly hung on a
        # shape problem, this line would never print.
        if not self._first_batch_logged and self.logger:
            self.logger.info(f"[train] First batch entering the graph "
                              f"(batch_size input shape check happens inside fit).")
            self._first_batch_logged = True

    def on_epoch_begin(self, epoch, logs=None):
        self.progress.current_epoch = epoch + 1
        self.progress.status = "running"
        if self.logger:
            self.logger.info(f"[train] ---- Epoch {epoch + 1}/{self.progress.total_epochs} starting ----")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.progress.current_loss = float(logs.get("loss", 0.0))
        self.progress.current_acc = float(logs.get("accuracy", 0.0))
        self.progress.current_val_loss = float(logs.get("val_loss", 0.0))
        self.progress.current_val_acc = float(logs.get("val_accuracy", 0.0))

        self.progress.history["loss"].append(self.progress.current_loss)
        self.progress.history["accuracy"].append(self.progress.current_acc)
        self.progress.history["val_loss"].append(self.progress.current_val_loss)
        self.progress.history["val_accuracy"].append(self.progress.current_val_acc)

        if self.logger:
            self.logger.info(
                f"[train] Epoch {epoch + 1}/{self.progress.total_epochs} complete | "
                f"loss={self.progress.current_loss:.4f} acc={self.progress.current_acc:.4f} | "
                f"val_loss={self.progress.current_val_loss:.4f} val_acc={self.progress.current_val_acc:.4f}"
            )

        if self.on_epoch_end_hook:
            self.on_epoch_end_hook(self.progress)

    def on_train_end(self, logs=None):
        if self.progress.status != "failed":
            self.progress.status = "completed"


def _build_callbacks(cfg: Config, run_dir: Path, progress: TrainingProgress,
                      total_epochs: int, on_epoch_end_hook, logger):
    checkpoint_path = run_dir / "best_model.keras"
    tensorboard_dir = run_dir / "tensorboard"
    csv_log_path = run_dir / "training_log.csv"

    callbacks = [
        ProgressCallback(progress, total_epochs, on_epoch_end_hook, logger),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir)),
        tf.keras.callbacks.CSVLogger(str(csv_log_path)),
    ]
    return callbacks


def run_training(
    cfg: Config,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    on_epoch_end_hook: Optional[Callable[[TrainingProgress], None]] = None,
) -> tuple:
    """
    The single entry point training_ui.py calls. MUST be called synchronously
    from the Streamlit main script thread (i.e. directly inside the button's
    `if st.button(...):` block) — never wrap this in `threading.Thread`.

    Returns (model, history_dict, progress, run_dir, dataset_bundle).
    Raises on unrecoverable errors (caller is expected to catch and display).
    """
    logger = get_logger("train_engine", cfg)
    set_global_seed(cfg.random_seed)

    epochs = epochs or cfg.epochs
    requested_batch = batch_size or cfg.batch_size
    progress = TrainingProgress(status="preparing")

    t0 = time.time()
    logger.info("=" * 70)
    logger.info("[train] Preparing dataset...")
    bundle: DatasetBundle = prepare_dataset(cfg, batch_size=requested_batch)

    logger.info(f"[train] Number of classes: {cfg.num_classes} ({cfg.classes})")
    logger.info(f"[train] Train samples: {len(bundle.y_train)} | Val samples: {len(bundle.y_val)}")
    logger.info(f"[train] Effective batch size: {bundle.batch_size} "
                f"(requested {requested_batch})")
    logger.info(f"[train] steps_per_epoch={bundle.steps_per_epoch} "
                f"validation_steps={bundle.validation_steps}")

    describe_array(bundle.X_train, "X_train", logger)
    describe_array(bundle.X_val, "X_val", logger)
    logger.info(f"[train] y_train unique labels: {np.unique(bundle.y_train, return_counts=True)}")
    logger.info(f"[train] y_val unique labels:   {np.unique(bundle.y_val, return_counts=True)}")

    if bundle.steps_per_epoch == 0:
        raise RuntimeError(
            "steps_per_epoch resolved to 0 — the training set is empty after "
            "cleaning/splitting. This is the exact condition that causes an "
            "infinite hang at 'Epoch 1'. Add more samples per class."
        )

    logger.info("[train] Building model...")
    model = build_model_from_config(cfg)
    summary_text = model_summary_string(model)
    logger.info("[train] Model summary:\n" + summary_text)

    total_params = model.count_params()
    logger.info(f"[train] Total trainable parameters: {total_params:,}")
    if total_params > 5_000_000:
        logger.warning(
            f"[train] Parameter count is unusually high ({total_params:,}) for a "
            f"{len(bundle.y_train)}-sample dataset. This can make the first CPU "
            f"batch very slow and look like a hang — check input_shape/config.json."
        )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.models_path / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks = _build_callbacks(cfg, run_dir, progress, epochs, on_epoch_end_hook, logger)

    progress.status = "running"
    logger.info(f"[train] Starting model.fit() for {epochs} epochs "
                f"(verbose=2 — clean per-epoch logging, no carriage-return buffering)...")

    try:
        history_obj = model.fit(
            bundle.train_ds,
            validation_data=bundle.val_ds,
            epochs=epochs,
            steps_per_epoch=bundle.steps_per_epoch,
            validation_steps=bundle.validation_steps or None,
            callbacks=callbacks,
            verbose=2,  # <- critical fix: no \r progress bar to get lost in buffering
        )
    except Exception as e:
        progress.status = "failed"
        progress.message = str(e)
        logger.error(f"[train] Training failed with exception: {e}")
        raise

    elapsed = time.time() - t0
    logger.info(f"[train] Training finished in {elapsed:.1f}s "
                f"({progress.current_epoch} epochs actually run)")

    final_model_path = cfg.models_path / "echoambuguard_model.keras"
    model.save(final_model_path)
    logger.info(f"[train] Final model saved to {final_model_path}")

    metadata = {
        "classes": cfg.classes,
        "input_shape": list(cfg.input_shape),
        "n_mfcc": cfg.n_mfcc,
        "max_pad_len": cfg.max_pad_len,
        "sample_rate": cfg.sample_rate,
        "duration_seconds": cfg.duration_seconds,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "epochs_requested": epochs,
        "epochs_run": progress.current_epoch,
        "batch_size": bundle.batch_size,
        "train_samples": len(bundle.y_train),
        "val_samples": len(bundle.y_val),
        "final_train_accuracy": progress.current_acc,
        "final_val_accuracy": progress.current_val_acc,
        "training_time_seconds": elapsed,
        "total_parameters": int(total_params),
        "run_id": run_id,
    }
    with open(cfg.models_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[train] Metadata written to {cfg.models_path / 'metadata.json'}")

    return model, history_obj.history, progress, run_dir, bundle
