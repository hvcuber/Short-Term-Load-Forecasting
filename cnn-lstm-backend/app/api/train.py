"""
POST /api/train
  Trains the CNN-LSTM model on the preprocessed tensors stored in app state.
  Requires /api/data/upload to have been called first.
"""
from fastapi import APIRouter, HTTPException, Request

from app.schemas.schemas import TrainRequest, TrainResponse
from app.services.model import model_service

router = APIRouter()


@router.post("", response_model=TrainResponse)
def train_model(body: TrainRequest, request: Request):
    tensors = getattr(request.app.state, "tensors", None)
    if tensors is None:
        raise HTTPException(
            status_code=400,
            detail="No data uploaded yet. Call POST /api/data/upload first.",
        )

    history = model_service.train(
        tensors["X_train"], tensors["y_train"],
        tensors["X_val"],   tensors["y_val"],
        epochs=body.epochs,
    )

    return TrainResponse(
        status="trained",
        epochs_trained=body.epochs,
        history=history,
    )
