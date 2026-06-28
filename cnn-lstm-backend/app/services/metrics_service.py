"""
All six evaluation metrics from the notebook:
MSE · MAE · RMSE · R² · MAPE · FADI
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.metrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
)


def compute_all(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_test_actual: np.ndarray,
    y_pred_actual: np.ndarray,
) -> dict:
    """
    y_test / y_pred        — normalised (0-1), used for MSE/MAE/RMSE/R²
    y_test_actual / y_pred_actual — inverse-transformed MW values, for MAPE & FADI
    """
    mse_m = MeanSquaredError()
    mse_m.update_state(y_test, y_pred)

    mae_m = MeanAbsoluteError()
    mae_m.update_state(y_test, y_pred)

    rmse_m = RootMeanSquaredError()
    rmse_m.update_state(y_test, y_pred)

    differences = (y_pred_actual - y_test_actual).astype(int).tolist()

    return {
        "mse":   float(mse_m.result().numpy()),
        "mae":   float(mae_m.result().numpy()),
        "rmse":  float(rmse_m.result().numpy()),
        "r2":    float(r2_score(y_test, y_pred) * 100),
        "mape":  float(_mape(y_test_actual, y_pred_actual)),
        "fadi":  float(_fadi(differences)),
    }


def _mape(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((actual - pred) / (actual + eps))) * 100)


def _fadi(differences: list[int]) -> float:
    mean = sum(differences) / len(differences)
    return sum(abs(d - mean) for d in differences) / len(differences)
