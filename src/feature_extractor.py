"""
feature_extractor.py
---------------------
Audio -> MFCC feature extraction, with every feature forced to an identical,
verified shape before it is ever written to disk. This is the single most
important defensive layer in the whole pipeline: if this file guarantees
every .npy feature is exactly (n_mfcc, max_pad_len) float32 with no NaN/Inf,
almost every downstream "hang" or "shape mismatch" bug becomes impossible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa

from src.utils import Config, get_logger, pad_or_truncate, sanitize_array, describe_array

logger = None  # initialised lazily via _log()


def _log(cfg: Config):
    global logger
    if logger is None:
        logger = get_logger("feature_extractor", cfg)
    return logger


# --------------------------------------------------------------------------- #
# Single-file extraction
# --------------------------------------------------------------------------- #
def load_audio(file_path: Path, cfg: Config) -> Optional[np.ndarray]:
    """
    Loads audio robustly. Tries soundfile first (fast, exact), falls back to
    librosa's loader (handles more codecs). Returns None (never raises) if
    the file is unreadable/corrupted so callers can skip it cleanly.
    """
    log = _log(cfg)
    try:
        data, sr = sf.read(str(file_path), always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # downmix to mono
        if sr != cfg.sample_rate:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=cfg.sample_rate)
        return data.astype(np.float32)
    except Exception as e_sf:
        try:
            data, sr = librosa.load(str(file_path), sr=cfg.sample_rate, mono=True)
            return data.astype(np.float32)
        except Exception as e_lr:
            log.error(f"[SKIP] Could not read audio '{file_path.name}': "
                      f"soundfile_error={e_sf} librosa_error={e_lr}")
            return None


def extract_mfcc(file_path: Path, cfg: Config) -> Optional[np.ndarray]:
    """
    Full single-file pipeline: load -> fix length -> MFCC -> pad/truncate ->
    sanitize. Always returns an array of EXACT shape (n_mfcc, max_pad_len)
    or None if the file could not be processed at all.
    """
    log = _log(cfg)
    audio = load_audio(file_path, cfg)
    if audio is None:
        return None

    if audio.size == 0:
        log.warning(f"[SKIP] Empty audio signal in '{file_path.name}'")
        return None

    target_len = int(cfg.sample_rate * cfg.duration_seconds)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    else:
        audio = audio[:target_len]

    try:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=cfg.sample_rate,
            n_mfcc=cfg.n_mfcc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
    except Exception as e:
        log.error(f"[SKIP] MFCC extraction failed for '{file_path.name}': {e}")
        return None

    mfcc = pad_or_truncate(mfcc, cfg.max_pad_len)          # guarantee exact shape
    mfcc = sanitize_array(mfcc, name=file_path.name, logger=log)  # guarantee no NaN/Inf

    if mfcc.shape != (cfg.n_mfcc, cfg.max_pad_len):
        log.error(f"[SKIP] Unexpected final shape {mfcc.shape} for '{file_path.name}'")
        return None

    return mfcc


# --------------------------------------------------------------------------- #
# Batch extraction over dataset_dir/<class_name>/*.wav
# --------------------------------------------------------------------------- #
SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def extract_and_save_all(cfg: Config) -> dict:
    """
    Walks dataset/<class>/*, extracts MFCCs, sanitizes them, and saves ONE
    consolidated .npy per class (features) plus a labels .npy, guaranteeing
    every saved feature has identical shape. Corrupted/unreadable files are
    skipped and reported, never allowed to silently corrupt the array.

    Returns a summary dict with per-class counts and skipped files, which
    training_ui.py surfaces directly in the Streamlit UI (requirement: show
    dataset statistics / class distribution).
    """
    log = _log(cfg)
    summary = {"classes": {}, "skipped": [], "total_saved": 0}

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []

    for class_idx, class_name in enumerate(cfg.classes):
        class_dir = cfg.dataset_path / class_name
        files = sorted(
            [p for p in class_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTS]
        )
        log.info(f"[extract] Class '{class_name}': found {len(files)} candidate audio files")

        saved_for_class = 0
        for fp in files:
            mfcc = extract_mfcc(fp, cfg)
            if mfcc is None:
                summary["skipped"].append(str(fp.name))
                continue
            all_features.append(mfcc)
            all_labels.append(class_idx)
            saved_for_class += 1

        summary["classes"][class_name] = saved_for_class

    if not all_features:
        log.error("[extract] No valid features extracted from ANY class. Aborting save.")
        return summary

    X = np.stack(all_features, axis=0).astype(np.float32)   # (N, n_mfcc, max_pad_len)
    y = np.array(all_labels, dtype=np.int64)

    describe_array(X, "X (all extracted features)", log)
    log.info(f"[extract] Label distribution: "
             f"{ {cfg.classes[i]: int((y == i).sum()) for i in range(cfg.num_classes)} }")

    np.save(cfg.features_path / "X_features.npy", X)
    np.save(cfg.features_path / "y_labels.npy", y)

    summary["total_saved"] = len(all_labels)
    log.info(f"[extract] Saved {summary['total_saved']} feature vectors to {cfg.features_path}")
    if summary["skipped"]:
        log.warning(f"[extract] Skipped {len(summary['skipped'])} corrupted/unreadable files: "
                    f"{summary['skipped']}")

    return summary


def features_exist(cfg: Config) -> bool:
    return (cfg.features_path / "X_features.npy").exists() and \
           (cfg.features_path / "y_labels.npy").exists()
