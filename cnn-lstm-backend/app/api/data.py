"""
POST /api/data/upload
  Accepts the two source files from the notebook:
    - load_file   : 2017-2022.xlsx
    - weather_file: 2017-01-01 to 2022-10-31.csv
  Saves them to data/ and runs the full preprocessing pipeline.
  The processed tensors are cached in app state for /train and /predict.
"""
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.schemas.schemas import DataUploadResponse
from app.services.preprocessing import PreprocessingService

router = APIRouter()
_svc   = PreprocessingService()


@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(
    request:      Request,
    load_file:    UploadFile = File(..., description="Load xlsx  (2017-2022.xlsx)"),
    weather_file: UploadFile = File(..., description="Weather CSV"),
):
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    load_path    = settings.DATA_DIR / "load_data.xlsx"
    weather_path = settings.DATA_DIR / "weather_data.csv"

    # stream to disk
    for upload, dest in [(load_file, load_path), (weather_file, weather_path)]:
        async with aiofiles.open(dest, "wb") as f:
            while chunk := await upload.read(1024 * 1024):
                await f.write(chunk)

    try:
        result = _svc.prepare(load_path, weather_path, save_scaler=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # store tensors in app state so train/predict routers can access them
    request.app.state.tensors = result

    return DataUploadResponse(
        status="ok",
        load_rows=len(result["final_df"]),
        weather_rows=len(result["final_df"]),
        message=(
            f"Preprocessed {len(result['final_df'])} rows. "
            f"Train: {result['X_train'].shape[0]}, "
            f"Val: {result['X_val'].shape[0]}, "
            f"Test: {result['X_test'].shape[0]} sequences."
        ),
    )
