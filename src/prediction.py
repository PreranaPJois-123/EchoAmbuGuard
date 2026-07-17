"""
prediction.py
--------------
Loads the trained model + metadata and runs inference on a single uploaded /
recorded audio file, using the EXACT same feature-extraction path as
training (src.feature_extractor.extract_mfcc) so there is never a train/
inference shape or preprocessing mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from src.utils import Config, get_logger
from src.feature_extractor import extract_mfcc


def load_trained_model(cfg: Config) -> Tuple[Optional[tf.keras.Model], Optional[dict]]:
    log = get_logger("prediction", cfg)
    model_path = cfg.models_path / "echoambuguard_model.keras"
    meta_path = cfg.models_path / "metadata.json"

    if not model_path.exists():
        log.warning("[predict] No trained model found. Train a model first.")
        return None, None

    model = tf.keras.models.load_model(model_path)

    metadata = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return model, metadata


def predict_file(file_path: Path, cfg: Config,
                  model: tf.keras.Model, metadata: Optional[dict] = None) -> dict:
    """
    Returns {"label": str, "confidence": float, "probabilities": {class: prob}}
    """
    log = get_logger("prediction", cfg)

    mfcc = extract_mfcc(file_path, cfg)
    if mfcc is None:
        raise ValueError(f"Could not extract features from '{file_path.name}' — "
                          f"the file may be corrupted or unreadable.")

    X = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)  # (1, n_mfcc, T, 1)
    log.info(f"[predict] Input shape for inference: {X.shape}")

    raw_pred = model.predict(X, verbose=0)

    if raw_pred.shape[-1] == 1:  # sigmoid / binary
        prob_positive = float(raw_pred[0][0])
        probs = {cfg.classes[0]: 1.0 - prob_positive, cfg.classes[1]: prob_positive}
        label = cfg.classes[1] if prob_positive >= 0.5 else cfg.classes[0]
        confidence = prob_positive if prob_positive >= 0.5 else 1.0 - prob_positive
    else:  # softmax / multiclass
        probs_arr = raw_pred[0]
        class_idx = int(np.argmax(probs_arr))
        label = cfg.classes[class_idx]
        confidence = float(probs_arr[class_idx])
        probs = {cfg.classes[i]: float(probs_arr[i]) for i in range(len(cfg.classes))}

    log.info(f"[predict] Prediction: {label} (confidence={confidence:.3f}) | probs={probs}")

    return {"label": label, "confidence": confidence, "probabilities": probs}
