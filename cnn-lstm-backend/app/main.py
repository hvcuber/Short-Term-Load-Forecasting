from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import data, predict, train, metrics
from app.core.config import settings

app = FastAPI(
    title="CNN-LSTM Load Forecasting API",
    description="BRPL SCADA Short-Term Load Forecasting — Hybrid CNN-LSTM Model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router,    prefix="/api/data",    tags=["Data"])
app.include_router(predict.router, prefix="/api/predict", tags=["Predict"])
app.include_router(train.router,   prefix="/api/train",   tags=["Train"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": settings.MODEL_PATH.exists()}
