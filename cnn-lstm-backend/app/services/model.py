"""
Model service — CNN-LSTM architecture from the notebook.
Handles build / train / save / load / predict.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, Flatten, LSTM, MaxPooling1D,
)
from tensorflow.keras.regularizers import L1

from app.core.config import settings


def build_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    """CNN-LSTM exactly as in the notebook."""
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        Conv1D(filters=32, kernel_size=2, strides=1,
               activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2, strides=1),
        Conv1D(filters=16, kernel_size=3, strides=1, activation="relu"),
        MaxPooling1D(pool_size=2, strides=1),
        LSTM(32),
        Flatten(),
        Dense(5, activation="relu"),
        Dropout(0.2),
        Dense(1, kernel_regularizer=L1(0.03)),
    ])
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer="adam",
        metrics=["mae"],
    )
    return model


class ModelService:
    _model: tf.keras.Model | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if settings.MODEL_PATH.exists():
            self._model = tf.keras.models.load_model(settings.MODEL_PATH)

    def save(self) -> None:
        if self._model is None:
            raise RuntimeError("No model to save.")
        settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(settings.MODEL_PATH)

    # ── training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        epochs:  int = settings.DEFAULT_EPOCHS,
    ) -> dict:
        tf.config.run_functions_eagerly(True)
        self._model = build_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
        history = self._model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=settings.BATCH_SIZE,
            verbose=0,
        )
        self.save()
        return {
            "loss":     history.history["loss"],
            "val_loss": history.history["val_loss"],
            "mae":      history.history["mae"],
            "val_mae":  history.history["val_mae"],
        }

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        return self._model.predict(X_test)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> tuple[float, float]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        loss, mae = self._model.evaluate(X_test, y_test, verbose=0)
        return float(loss), float(mae)

    def summary(self) -> list[str]:
        if self._model is None:
            return []
        lines: list[str] = []
        self._model.summary(print_fn=lambda x: lines.append(x))
        return lines


# singleton — imported by routers
model_service = ModelService()
