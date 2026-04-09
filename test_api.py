from fastapi.testclient import TestClient
from app import app
import yaml

client = TestClient(app)

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def test_forecast_horizon_length():
    config = load_config()
    horizon = config["model"]["horizons"][0]

    response = client.get(f"/forecast?horizon={horizon}")
    assert response.status_code == 200

    data = response.json()["forecast"]
    assert len(data) > 0

    from collections import Counter
    counts = Counter([row["unique_id"] for row in data])

    for count in counts.values():
        assert count == horizon