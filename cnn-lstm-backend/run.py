#!/usr/bin/env python3
"""
Run this instead of uvicorn directly if you want the model
pre-loaded from disk before the first request arrives.
Drop-in replacement: `python run.py`
"""
import uvicorn
from app.services.model import model_service

if __name__ == "__main__":
    # pre-load saved model (no-op if saved_models/cnn_lstm.keras doesn't exist yet)
    model_service.load()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
