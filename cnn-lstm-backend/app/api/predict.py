"""
POST /api/predict
  Runs inference on test tensors and returns actual vs predicted SCADA load
  with OD/UD labels — exactly as computed in the notebook's comp_df.
"""
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.schemas.schemas import PredictResponse, PredictionPoint
from app.services.model import model_service
from app.services.preprocessing import PreprocessingService

router = APIRouter()
_svc   = PreprocessingService()


@router.post("", response_model=PredictResponse)
def predict(request: Request):
    tensors = getattr(request.app.state, "tensors", None)
    if tensors is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet.")

    if not model_service.is_loaded():
        raise HTTPException(status_code=400, detail="Model not trained or loaded.")

    X_test = tensors["X_test"]
    y_test = tensors["y_test"]
    scaler = _svc.load_scaler()
    dates  = tensors["test_dates"]

    y_pred = model_service.predict(X_test)          # shape (N, 1)

    # inverse-transform: repeat single column to match scaler's 14-feature input
    n_features = scaler.n_features_in_

    def inverse(arr_1d: np.ndarray) -> np.ndarray:
        tiled = np.repeat(arr_1d.reshape(-1, 1), n_features, axis=-1)
        return scaler.inverse_transform(tiled)[:, -1]

    actual_mw    = inverse(y_test).astype(int)
    predicted_mw = inverse(y_pred.flatten()).astype(int)
    differences  = predicted_mw - actual_mw

    points: list[PredictionPoint] = []
    for i, (dt, act, pred, diff) in enumerate(
        zip(dates, actual_mw, predicted_mw, differences)
    ):
        points.append(
            PredictionPoint(
                datetime=str(dt),
                actual_mw=float(act),
                predicted_mw=float(pred),
                difference=int(diff),
                od_ud=f"{'+' if diff >= 0 else ''}{diff}",
            )
        )

    return PredictResponse(
        status="ok",
        total_points=len(points),
        predictions=points,
    )
