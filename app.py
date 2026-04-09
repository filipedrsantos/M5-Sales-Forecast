import os
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MSE

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="M5 Sales Forecast API")

CONFIG_FILE = "config.yaml"

# -------------------------------
# Input model for /forecast
# -------------------------------
class ForecastRequest(BaseModel):
    horizon: int = 30
    model_version: Optional[str] = None
    unique_ids: Optional[List[str]] = None
    top_k: Optional[int] = None

# -------------------------------
# Load configuration
# -------------------------------
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data"]["processed_path"]
MODELS_DIR = config["paths"]["models_dir"]

HORIZONS = config["model"]["horizons"]
INPUT_SIZE = config["model"]["input_size"]
MAX_STEPS = config["model"]["max_steps"]
FREQ = config["model"]["freq"]

# -------------------------------
# Utility functions
# -------------------------------
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def build_nhits(horizon):
    return NHITS(
        h=horizon,
        input_size=INPUT_SIZE,
        max_steps=2000,
        loss=MSE(),
        scaler_type="standard",
        futr_exog_list=[
            "is_weekend",
            "snap",
            "sell_price",
            "price_change",
            "event_flag",
            "day_of_week",
            "month"
        ]
    )

def naive_forecast(train_df, val_df):
    naive_preds = (
        train_df.sort_values("ds")
        .groupby("unique_id")
        .tail(1)[["unique_id", "y"]]
        .rename(columns={"y":"naive_pred"})
    )
    val_naive = val_df.merge(naive_preds, on="unique_id", how="left")
    val_naive["naive_pred"] = val_naive.groupby("unique_id")["naive_pred"].transform("first")
    return val_naive

def evaluate_forecast(val_df, merged, horizon_name="HORIZON"):
    y_true = merged["y"].values
    y_pred = merged["NHITS"].values
    val_smape = smape(y_true, y_pred)
    val_rmse = rmse(y_true, y_pred)
    
    metrics_per_product = merged.groupby("unique_id").apply(
        lambda x: pd.Series({
            "sMAPE": smape(x["y"], x["NHITS"]),
            "RMSE": rmse(x["y"], x["NHITS"])
        })
    ).reset_index()
    metrics_per_product["horizon"] = horizon_name
    return val_smape, val_rmse, metrics_per_product

def get_latest_model_version():
    versions = [f for f in os.listdir(MODELS_DIR) if f.startswith("nhits_")]
    if not versions:
        raise HTTPException(status_code=404, detail="No trained model found")
    return sorted(versions)[-1]

# -------------------------------
# POST /train
# -------------------------------
@app.post("/train")
def train_models():
    df = pd.read_csv(DATA_PATH)
    df["ds"] = pd.to_datetime(df["ds"])
    df["unique_id"] = df["unique_id"].astype(str)

    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(MODELS_DIR, f"nhits_{version_id}")
    os.makedirs(model_dir, exist_ok=True)

    metrics_all = []

    for idx, horizon in enumerate(HORIZONS):
        cutoff = df["ds"].max() - pd.Timedelta(days=horizon)
        train_df = df[df["ds"] <= cutoff].copy()
        val_df = df[df["ds"] > cutoff].copy()

        # -------------------------------
        # Train model (IMPROVED)
        # -------------------------------
        from neuralforecast.losses.pytorch import MAE

        model = NHITS(
            h=horizon,
            input_size=INPUT_SIZE,
            max_steps=1500,
            loss=MAE(),
            scaler_type="standard",
            futr_exog_list=[
                "sell_price",
                "price_change",
                "lag_1",
                "lag_7",
                "was_zero_yesterday"
            ]
        )

        nf = NeuralForecast(models=[model], freq=FREQ)
        nf.fit(df=train_df)

        # -------------------------------
        # Save model
        # -------------------------------
        horizon_dir = model_dir if idx == 0 else os.path.join(model_dir, f"h_{horizon}")
        os.makedirs(horizon_dir, exist_ok=True)
        nf.save(path=horizon_dir)

        # -------------------------------
        # Forecast
        # -------------------------------
        futr_df = val_df[[
            "unique_id",
            "ds",
            "sell_price",
            "price_change",
            "lag_1",
            "lag_7",
            "was_zero_yesterday"
        ]].copy()

        forecasts = nf.predict(futr_df=futr_df)

        # Remove negative predictions
        forecasts["NHITS"] = forecasts["NHITS"].clip(lower=0)

        # Optional cap (avoid spikes)
        forecasts["NHITS"] = forecasts["NHITS"].clip(upper=50)

        # Rounding
        forecasts["y_pred_rounded"] = forecasts["NHITS"].round()
        forecasts.loc[forecasts["y_pred_rounded"] < 1, "y_pred_rounded"] = 0
        forecasts["y_pred_rounded"] = forecasts["y_pred_rounded"].astype(int)

        # -------------------------------
        # Merge with actuals and naive
        # -------------------------------
        naive_df = naive_forecast(train_df, val_df)

        merged = forecasts.merge(
            val_df[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left"
        )

        merged = merged.merge(
            naive_df[["unique_id", "ds", "naive_pred"]],
            on=["unique_id", "ds"],
            how="left"
        )

        merged["horizon"] = f"{horizon}d"

        # -------------------------------
        # Metrics
        # -------------------------------
        val_smape, val_rmse, metrics_per_product = evaluate_forecast(
            val_df, merged, f"{horizon}d"
        )

        metrics_all.append((metrics_per_product, val_smape, val_rmse))

        # -------------------------------
        # Save CSVs
        # -------------------------------
        columns_order = [
            "unique_id",
            "ds",
            "y",
            "NHITS",
            "y_pred_rounded",
            "naive_pred"
        ]

        merged = merged[columns_order]

        merged.to_csv(
            os.path.join(horizon_dir, "forecast_vs_actual.csv"),
            index=False
        )

        metrics_per_product.to_csv(
            os.path.join(horizon_dir, "metrics_per_product.csv"),
            index=False
        )

    # -------------------------------
    # Save training summary
    # -------------------------------
    cfg_summary = {
        "model": "NHITS",
        "version_id": version_id,
        "horizons": HORIZONS,
        "input_size": INPUT_SIZE,
        "max_steps": 1500,
        "features_used": ["sell_price", "price_change"],
        "metrics_summary": {
            f"{horizon}d": {
                "sMAPE": float(val_smape),
                "RMSE": float(val_rmse)
            }
            for (_, val_smape, val_rmse), horizon in zip(metrics_all, HORIZONS)
        }
    }

    with open(os.path.join(model_dir, "training_config.json"), "w") as f:
        json.dump(cfg_summary, f, indent=4)

    return {
        "message": "Training completed",
        "version_id": version_id
    }

# -------------------------------
# GET /forecast
# -------------------------------
@app.get("/forecast")
def forecast(
    horizon: int = Query(30),
    model_version: Optional[str] = None,
    unique_ids: Optional[List[str]] = Query(None),
    top_k: Optional[int] = None
):
    version = model_version or get_latest_model_version()
    model_dir = os.path.join(MODELS_DIR, version)

    # Select horizon folder
    horizon_dir = model_dir if horizon == HORIZONS[0] else os.path.join(model_dir, f"h_{horizon}")
    forecast_file = os.path.join(horizon_dir, "forecast_vs_actual.csv")
    if not os.path.exists(forecast_file):
        raise HTTPException(status_code=404, detail="Forecast file not found")

    df_forecast = pd.read_csv(forecast_file)

    # Filter specific series
    if unique_ids:
        df_forecast = df_forecast[df_forecast["unique_id"].isin(unique_ids)]

    # Filter top-K by predicted sales
    if top_k:
        top_series = (
            df_forecast.groupby("unique_id")["NHITS"]
            .sum()
            .sort_values(ascending=False)
            .head(top_k)
            .index
        )
        df_forecast = df_forecast[df_forecast["unique_id"].isin(top_series)]

    return {
        "model_version": version,
        "forecast": df_forecast.to_dict(orient="records")
    }
# -------------------------------
# GET /health
# -------------------------------
@app.get("/health")
def health():
    return {"status":"ok","time":datetime.now().isoformat()}