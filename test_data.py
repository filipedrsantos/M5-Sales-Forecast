import pandas as pd
import yaml

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def test_processed_schema():
    config = load_config()
    path = config["data"]["processed_path"]

    df = pd.read_csv(path)

    assert set(["unique_id", "ds", "y"]).issubset(df.columns)

def test_monotonic_dates():
    config = load_config()
    path = config["data"]["processed_path"]

    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])

    grouped = df.groupby("unique_id")

    for _, g in grouped:
        assert g["ds"].is_monotonic_increasing