import pandas as pd
import yaml

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def test_no_leakage_split():
    config = load_config()
    path = config["data"]["processed_path"]
    horizons = config["model"]["horizons"]

    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])

    horizon = max(horizons)
    cutoff = df["ds"].max() - pd.Timedelta(days=horizon)

    train_df = df[df["ds"] <= cutoff]
    val_df = df[df["ds"] > cutoff]

    assert train_df["ds"].max() < val_df["ds"].min()