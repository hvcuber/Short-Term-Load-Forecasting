from __future__ import annotations
from pydantic import BaseModel, Field


# ── /api/train ────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int = Field(default=10, ge=1, le=200)


class TrainResponse(BaseModel):
    status: str
    epochs_trained: int
    history: dict[str, list[float]]   # loss, val_loss, mae, val_mae


# ── /api/predict ──────────────────────────────────────────────────────────────

class PredictionPoint(BaseModel):
    datetime: str
    actual_mw: float
    predicted_mw: float
    difference: int
    od_ud: str                         # "+123" or "-45"


class PredictResponse(BaseModel):
    status: str
    total_points: int
    predictions: list[PredictionPoint]


# ── /api/metrics ──────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    mse:  float
    mae:  float
    rmse: float
    r2:   float    # percentage
    mape: float    # percentage
    fadi: float


# ── /api/data ─────────────────────────────────────────────────────────────────

class DataUploadResponse(BaseModel):
    status: str
    load_rows: int
    weather_rows: int
    message: str
