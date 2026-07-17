"""
model_builder.py
------------------
Builds the CNN used for ambulance-siren classification.

Design choice that matters for your bug specifically: this architecture uses
GlobalAveragePooling2D instead of Flatten -> Dense(large). On a tiny dataset
with inconsistent padding, a Flatten layer can silently produce a Dense
layer with hundreds of thousands (or millions) of parameters, and the very
first CPU training step can then take an abnormally long time — which is
indistinguishable from a "hang". GlobalAveragePooling2D collapses the
spatial dimensions to a fixed, small size regardless of any upstream shape
drift, making the model both robust and fast to train.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from src.utils import Config


def build_cnn(input_shape: Tuple[int, int, int], num_classes: int, learning_rate: float) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="mfcc_input")

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)  # <- fixed-size output regardless of input drift
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
        loss = "sparse_categorical_crossentropy"

    model = models.Model(inputs, outputs, name="EchoAmbuGuard_CNN")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def build_model_from_config(cfg: Config) -> tf.keras.Model:
    return build_cnn(cfg.input_shape, cfg.num_classes, cfg.learning_rate)


def model_summary_string(model: tf.keras.Model) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.summary()
    return buf.getvalue()
