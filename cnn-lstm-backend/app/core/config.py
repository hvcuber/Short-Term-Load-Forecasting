from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_PATH: Path = BASE_DIR / "saved_models" / "cnn_lstm.keras"
    SCALER_PATH: Path = BASE_DIR / "saved_models" / "scaler.pkl"

    # Model hyperparameters (match your notebook)
    LOOK_BACK: int = 96          # 96 × 15-min slots = 1 day
    N_FEATURES: int = 13         # all features except target
    TRAIN_SPLIT: float = 0.70
    VAL_SPLIT: float = 0.10
    TEST_SPLIT: float = 0.20
    BATCH_SIZE: int = 100
    DEFAULT_EPOCHS: int = 10

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:80"]

    class Config:
        env_file = ".env"


settings = Settings()
