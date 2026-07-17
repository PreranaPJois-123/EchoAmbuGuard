"""
dataset_manager.py
-------------------
Loads saved features, validates/cleans them, performs a safe stratified
train/val split (safe even with as few as 5 samples per class), and builds
tf.data.Dataset pipelines with batching/prefetch/AUTOTUNE.

This module is where two of the four hang-hypotheses are neutralised:
  1. Shape mismatches / corrupted feature files -> validated + skipped here.
  2. batch_size larger than the validation split -> clamped here so Keras
     is never left waiting on a validation batch that can't be filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.utils import Config, get_logger, sanitize_array, describe_array


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    batch_size: int
    class_counts: dict
    steps_per_epoch: int
    validation_steps: int


def load_raw_features(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    log = get_logger("dataset_manager", cfg)
    X_path = cfg.features_path / "X_features.npy"
    y_path = cfg.features_path / "y_labels.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Feature files not found. Run feature extraction before training."
        )

    try:
        X = np.load(X_path)
        y = np.load(y_path)
    except Exception as e:
        raise RuntimeError(f"Feature files are corrupted and could not be loaded: {e}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature/label count mismatch: X has {X.shape[0]} samples, "
            f"y has {y.shape[0]} labels."
        )

    log.info(f"[load] Loaded raw features: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def validate_and_clean(X: np.ndarray, y: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforces exact shape (N, n_mfcc, max_pad_len), drops any sample whose
    shape is wrong beyond automatic fixing, cleans NaN/Inf, and reports full
    diagnostics — covers requirement #3 (verify shapes/labels/dtype/NaN/Inf).
    """
    log = get_logger("dataset_manager", cfg)

    expected_feat_shape = (cfg.n_mfcc, cfg.max_pad_len)
    keep_idx = []
    for i in range(X.shape[0]):
        sample = X[i]
        if sample.shape != expected_feat_shape:
            log.warning(f"[clean] Dropping sample {i}: shape {sample.shape} != "
                        f"expected {expected_feat_shape}")
            continue
        keep_idx.append(i)

    if len(keep_idx) != X.shape[0]:
        X = X[keep_idx]
        y = y[keep_idx]

    X = sanitize_array(X, name="X_features", logger=log)
    y = np.asarray(y, dtype=np.int64)

    unique_labels = np.unique(y)
    if unique_labels.min() < 0 or unique_labels.max() >= cfg.num_classes:
        raise ValueError(
            f"Labels out of range for {cfg.num_classes} classes: found {unique_labels}"
        )

    describe_array(X, "X_features (cleaned)", log)
    class_counts = {cfg.classes[i]: int((y == i).sum()) for i in range(cfg.num_classes)}
    log.info(f"[clean] Final class distribution: {class_counts}")
    log.info(f"[clean] Number of classes: {cfg.num_classes} | Total samples: {len(y)}")

    for cls, count in class_counts.items():
        if count == 0:
            raise ValueError(
                f"Class '{cls}' has 0 valid samples after cleaning — cannot train."
            )

    return X, y


def safe_stratified_split(
    X: np.ndarray, y: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a stratified split that degrades gracefully on tiny datasets
    (e.g. 5 samples/class). If stratification or a normal split would leave
    a class with zero validation samples, falls back to taking exactly
    `min_val_samples_per_class` samples per class for validation manually.
    """
    log = get_logger("dataset_manager", cfg)
    n_total = len(y)
    n_val_target = max(
        cfg.num_classes * cfg.min_val_samples_per_class,
        int(round(n_total * cfg.validation_split)),
    )

    if n_total - n_val_target < cfg.num_classes:
        log.warning(
            "[split] Dataset too small for a proportional split — "
            "falling back to manual per-class split."
        )
        train_idx, val_idx = [], []
        for cls_idx in range(cfg.num_classes):
            cls_positions = np.where(y == cls_idx)[0]
            n_val = min(cfg.min_val_samples_per_class, max(1, len(cls_positions) // 5))
            n_val = max(1, n_val) if len(cls_positions) > 1 else 0
            val_idx.extend(cls_positions[:n_val])
            train_idx.extend(cls_positions[n_val:])
        train_idx, val_idx = np.array(train_idx), np.array(val_idx)
        if len(val_idx) == 0:  # absolute last resort: reuse train as val
            val_idx = train_idx
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=cfg.validation_split,
            random_state=cfg.random_seed,
            stratify=y,
        )

    log.info(f"[split] Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")
    return X_train, y_train, X_val, y_val


def build_tf_datasets(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: Config, batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Builds tf.data.Dataset pipelines with shuffle/batch/prefetch/AUTOTUNE.

    Critically: batch_size is clamped so it can NEVER exceed the number of
    training or validation samples, and drop_remainder is left False, so
    steps_per_epoch/validation_steps are always >= 1. This directly prevents
    the "0 steps -> fit() waits forever" failure mode.
    """
    log = get_logger("dataset_manager", cfg)

    X_train = X_train[..., np.newaxis].astype(np.float32)  # add channel dim
    X_val = X_val[..., np.newaxis].astype(np.float32)

    effective_batch = max(1, min(batch_size, len(X_train), len(X_val) or 1))
    if effective_batch != batch_size:
        log.warning(f"[dataset] Requested batch_size={batch_size} is larger than the "
                     f"dataset allows — clamped to {effective_batch}.")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=max(len(X_train), 1), seed=cfg.random_seed, reshuffle_each_iteration=True)
        .batch(effective_batch, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(effective_batch, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    steps_per_epoch = int(np.ceil(len(X_train) / effective_batch))
    validation_steps = int(np.ceil(len(X_val) / effective_batch)) if len(X_val) else 0

    log.info(f"[dataset] effective_batch_size={effective_batch} "
             f"steps_per_epoch={steps_per_epoch} validation_steps={validation_steps}")

    return train_ds, val_ds, effective_batch, steps_per_epoch


def prepare_dataset(cfg: Config, batch_size: int | None = None) -> DatasetBundle:
    """Single entry point used by training_ui.py / train_engine.py."""
    log = get_logger("dataset_manager", cfg)
    batch_size = batch_size or cfg.batch_size

    X, y = load_raw_features(cfg)
    X, y = validate_and_clean(X, y, cfg)
    X_train, y_train, X_val, y_val = safe_stratified_split(X, y, cfg)

    train_ds, val_ds, effective_batch, steps_per_epoch = build_tf_datasets(
        X_train, y_train, X_val, y_val, cfg, batch_size
    )
    validation_steps = int(np.ceil(len(X_val) / effective_batch)) if len(X_val) else 0

    class_counts = {cfg.classes[i]: int((y == i).sum()) for i in range(cfg.num_classes)}

    log.info("[prepare] Dataset preparation complete.")
    return DatasetBundle(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        train_ds=train_ds, val_ds=val_ds,
        batch_size=effective_batch,
        class_counts=class_counts,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )


def dataset_statistics(cfg: Config) -> dict:
    """
    Lightweight stats used by the Streamlit UI BEFORE training starts
    (dataset statistics / class distribution / feature dimensions), without
    needing to build tf.data pipelines.
    """
    log = get_logger("dataset_manager", cfg)
    try:
        X, y = load_raw_features(cfg)
    except FileNotFoundError:
        return {"available": False}

    class_counts = {cfg.classes[i]: int((y == i).sum()) for i in range(cfg.num_classes)}
    return {
        "available": True,
        "total_samples": int(len(y)),
        "class_counts": class_counts,
        "feature_shape": tuple(X.shape[1:]),
        "dtype": str(X.dtype),
    }
