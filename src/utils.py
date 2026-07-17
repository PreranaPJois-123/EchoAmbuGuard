"""
utils.py
--------
Shared, dependency-light utilities used by every stage of the EchoAmbuGuard
pipeline: config loading, logging setup, reproducibility, and the array
sanitation helpers that make the training pipeline robust against corrupted
features, NaNs, Infs, and shape mismatches.

Every function here is intentionally pure / side-effect-light so it can be
unit tested and reused identically from feature_extractor.py,
dataset_manager.py, train_engine.py, prediction.py and training_ui.py.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    sample_rate: int
    duration_seconds: float
    n_mfcc: int
    max_pad_len: int
    n_fft: int
    hop_length: int

    dataset_dir: str
    features_dir: str
    models_dir: str
    logs_dir: str

    classes: list

    batch_size: int
    max_batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    min_val_samples_per_class: int
    random_seed: int

    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    min_learning_rate: float

    @property
    def dataset_path(self) -> Path:
        return PROJECT_ROOT / self.dataset_dir

    @property
    def features_path(self) -> Path:
        return PROJECT_ROOT / self.features_dir

    @property
    def models_path(self) -> Path:
        return PROJECT_ROOT / self.models_dir

    @property
    def logs_path(self) -> Path:
        return PROJECT_ROOT / self.logs_dir

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        # (n_mfcc, time_frames, channels)
        return (self.n_mfcc, self.max_pad_len, 1)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"config.json not found at {path}")
    with open(path, "r") as f:
        raw: Dict[str, Any] = json.load(f)
    cfg = Config(**raw)

    for p in (cfg.dataset_path, cfg.features_path, cfg.models_path, cfg.logs_path):
        p.mkdir(parents=True, exist_ok=True)
    for cls in cfg.classes:
        (cfg.dataset_path / cls).mkdir(parents=True, exist_ok=True)

    return cfg


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, cfg: Config | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger that writes to both stdout (with explicit flush so it
    NEVER appears to 'hang' behind a buffered terminal) and a rotating-free
    plain log file under logs/.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on Streamlit re-runs
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.flush = sys.stdout.flush  # force flush every record
    logger.addHandler(stream_handler)

    if cfg is not None:
        log_file = cfg.logs_path / "pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Array sanitation — THIS is what makes the training pipeline robust
# --------------------------------------------------------------------------- #
def pad_or_truncate(mfcc: np.ndarray, max_pad_len: int) -> np.ndarray:
    """
    Force an MFCC array of shape (n_mfcc, T) into shape (n_mfcc, max_pad_len)
    by truncating or zero-padding along the time axis.

    This is the single most common source of a "hang" in small CNN + Streamlit
    projects: if even one saved feature has a different T, np.array(list_of_
    features) produces a ragged / object array (or, in newer NumPy, raises).
    Downstream this either crashes Keras with a cryptic error, or — if it
    silently broadcasts — creates a Flatten layer with a wildly different
    parameter count than expected, causing the *first* training step to take
    an abnormally long time on CPU (which looks exactly like a hang).
    """
    if mfcc.ndim != 2:
        raise ValueError(f"Expected 2D MFCC (n_mfcc, T), got shape {mfcc.shape}")

    n_mfcc, t = mfcc.shape
    if t == max_pad_len:
        return mfcc
    if t > max_pad_len:
        return mfcc[:, :max_pad_len]
    pad_width = max_pad_len - t
    return np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")


def sanitize_array(
    arr: np.ndarray,
    name: str = "array",
    logger: logging.Logger | None = None,
    nan_fill: float = 0.0,
    clip_value: float = 1e6,
) -> np.ndarray:
    """
    Replace NaN -> nan_fill, +Inf/-Inf -> +/-clip_value. Returns a new,
    dtype=float32 array. Logs how many bad values were found so the user
    can see exactly what was wrong with their data (per your requirement #4).
    """
    arr = np.asarray(arr, dtype=np.float32)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    if nan_count or inf_count:
        if logger:
            logger.warning(
                f"[sanitize] {name}: found {nan_count} NaN and {inf_count} Inf "
                f"values out of {arr.size} — cleaning."
            )
        arr = np.nan_to_num(
            arr, nan=nan_fill, posinf=clip_value, neginf=-clip_value
        )
    return arr


def describe_array(arr: np.ndarray, name: str, logger: logging.Logger) -> None:
    """Prints the full diagnostic block requested: shape/min/max/nan/inf/dtype."""
    arr_f = np.asarray(arr, dtype=np.float64)
    logger.info(
        f"[debug] {name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={np.nanmin(arr_f):.4f} max={np.nanmax(arr_f):.4f} "
        f"mean={np.nanmean(arr_f):.4f} "
        f"nan_count={int(np.isnan(arr_f).sum())} "
        f"inf_count={int(np.isinf(arr_f).sum())}"
    )
