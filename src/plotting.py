"""
plotting.py
------------
Matplotlib figure builders for the Streamlit training page: accuracy curve,
loss curve, and confusion matrix. All functions return a Figure object
(never call plt.show()) so they compose cleanly with st.pyplot().
"""

from __future__ import annotations

from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_training_curves(history: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.get("accuracy", []), label="Train Accuracy", marker="o")
    axes[0].plot(history.get("val_accuracy", []), label="Val Accuracy", marker="o")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.get("loss", []), label="Train Loss", marker="o", color="tomato")
    axes[1].plot(history.get("val_loss", []), label="Val Loss", marker="o", color="darkred")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int],
                           class_names: Sequence[str]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
