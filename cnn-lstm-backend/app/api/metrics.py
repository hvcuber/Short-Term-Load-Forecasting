"""
GET /api/metrics
  Returns MSE, MAE, RMSE, R², MAPE, FADI for the test set.
"""
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.schemas.schemas import MetricsResponse
from app.services.metrics_service import compute_all
from app.services.model import model_service
from app.services.preprocessing import PreprocessingService

router = APIRouter()
_svc   = PreprocessingService()


@router.get("", response_model=MetricsResponse)
def get_metrics(request: Request):
    tensors = getattr(request.app.state, "tensors", None)
    if tensors is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet.")

    if not model_service.is_loaded():
        raise HTTPException(status_code=400, detail="Model not trained or loaded.")

    X_test = tensors["X_test"]
    y_test = tensors["y_test"]
    scaler = _svc.load_scaler()

    y_pred = model_service.predict(X_test).flatten()

    n = scaler.n_features_in_

    def inverse(arr: np.ndarray) -> np.ndarray:
        return scaler.inverse_transform(
            np.repeat(arr.reshape(-1, 1), n, axis=-1)
        )[:, -1]

    actual_mw    = inverse(y_test)
    predicted_mw = inverse(y_pred)

    result = compute_all(y_test, y_pred, actual_mw, predicted_mw)
    return MetricsResponse(**result)
