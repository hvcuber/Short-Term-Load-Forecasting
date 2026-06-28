"""
Preprocessing service — mirrors the notebook pipeline exactly:
  load xlsx  →  spline interpolation  →  merge weather CSV  →
  feature engineering  →  train/val/test split  →  MinMaxScaler
"""
from __future__ import annotations

import pickle
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.core.config import settings


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")

    # spline interpolation for missing values (order-5, both directions)
    for col_raw, col_new in [
        ("demand load", "DEMAND LOAD"),
        ("sldc load",   "SLDC LOAD"),
        ("scada load",  "SCADA LOAD"),
    ]:
        df[col_new] = df[col_raw].interpolate(
            method="spline", order=5, limit_direction="both"
        )

    out = pd.DataFrame(
        {
            "datetime":    df["datetime"],
            "DEMAND LOAD": df["DEMAND LOAD"],
            "SLDC LOAD":   df["SLDC LOAD"],
            "SCADA LOAD":  df["SCADA LOAD"],
        }
    )
    out.index = pd.to_datetime(out["datetime"].astype(str), format="%d-%m-%Y %H:%M:%S")
    out = out.rename_axis("datetime").reset_index(drop=True)
    return out


def _load_weather_data(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(path)
    hourly = pd.DataFrame(
        {"temp": weather["temp"], "humidity": weather["humidity"]}
    )
    hourly.index = pd.to_datetime(weather["datetime"])

    temp     = hourly["temp"].resample("15Min").mean().interpolate(
        method="spline", order=5, limit_direction="both"
    )
    humidity = hourly["humidity"].resample("15Min").mean().interpolate(
        method="spline", order=5, limit_direction="both"
    )

    out = pd.DataFrame({"temp": temp, "humidity": humidity}).iloc[:-1].reset_index()
    return out


def _build_features(load_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "DATETIME":     load_df["datetime"],
            "TEMPERATURE":  weather_df["temp"].values,
            "HUMIDITY":     weather_df["humidity"].values,
            "DEMAND LOAD":  load_df["DEMAND LOAD"].values,
            "SLDC LOAD":    load_df["SLDC LOAD"].values,
            "SCADA LOAD":   load_df["SCADA LOAD"].values,
        }
    )
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])

    # holiday flag
    years       = df["DATETIME"].dt.year.unique().tolist()
    holiday_set = set(holidays.India(years=years).keys())
    df["IS_HOLIDAY"] = df["DATETIME"].dt.date.apply(
        lambda d: 1 if d in holiday_set else 0
    )

    # calendar features
    df["YEAR"]        = df["DATETIME"].dt.year
    df["MONTH"]       = df["DATETIME"].dt.month
    df["DAY"]         = df["DATETIME"].dt.day
    df["DAY_OF_YEAR"] = df["DATETIME"].dt.dayofyear
    df["WEEK_OF_YEAR"]= df["DATETIME"].dt.isocalendar().week.astype(int)
    df["QUARTER"]     = df["DATETIME"].dt.quarter
    df["SEASON"]      = df["MONTH"] % 12 // 3 + 1

    # keep last 3 years (matching notebook)
    cutoff = df.loc[df["DATETIME"] == "2019-10-31 00:00:00"].index
    if len(cutoff):
        df = df.loc[cutoff[0]:].reset_index(drop=True)

    cols = [
        "DATETIME", "YEAR", "MONTH", "DAY", "DAY_OF_YEAR",
        "WEEK_OF_YEAR", "QUARTER", "SEASON", "IS_HOLIDAY",
        "TEMPERATURE", "HUMIDITY", "DEMAND LOAD", "SLDC LOAD", "SCADA LOAD",
    ]
    return df[cols]


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n         = len(df)
    train_end = int(n * settings.TRAIN_SPLIT)
    val_end   = train_end + int(n * settings.VAL_SPLIT)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _create_sequences(
    data: np.ndarray, labels: np.ndarray, look_back: int
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(look_back, data.shape[0] + 1):
        X.append(data[i - look_back : i])
        y.append(labels[i - 1])
    return np.array(X), np.array(y)


# ── public API ────────────────────────────────────────────────────────────────

class PreprocessingService:
    """Stateless preprocessing — call prepare() to get train/val/test tensors."""

    def prepare(
        self,
        load_path: Path,
        weather_path: Path,
        save_scaler: bool = True,
    ) -> dict:
        load_df    = _load_load_data(load_path)
        weather_df = _load_weather_data(weather_path)
        final      = _build_features(load_df, weather_df)

        train_df, val_df, test_df = _split(final)

        scaler = MinMaxScaler(feature_range=(0, 1))

        def scale(df: pd.DataFrame) -> np.ndarray:
            arr = df.drop(columns=["DATETIME"]).values
            return scaler.fit_transform(arr)

        s_train = scale(train_df)
        s_val   = scale(val_df)
        s_test  = scale(test_df)

        lb = settings.LOOK_BACK
        X_train, y_train = _create_sequences(s_train[:, :-1], s_train[:, -1], lb)
        X_val,   y_val   = _create_sequences(s_val[:,   :-1], s_val[:,   -1], lb)
        X_test,  y_test  = _create_sequences(s_test[:,  :-1], s_test[:,  -1], lb)

        if save_scaler:
            settings.SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(settings.SCALER_PATH, "wb") as f:
                pickle.dump(scaler, f)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
            "scaler":  scaler,
            "test_start_date": test_df["DATETIME"].iloc[lb],
            "test_dates":      test_df["DATETIME"].iloc[lb:].reset_index(drop=True),
            "final_df":        final,
        }

    def load_scaler(self) -> MinMaxScaler:
        with open(settings.SCALER_PATH, "rb") as f:
            return pickle.load(f)
