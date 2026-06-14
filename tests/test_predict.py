"""
tests/test_predict.py
------------------------
Unit test for the /predict endpoint (Part 1 / Section 9.3.3).

Uses FastAPI's TestClient. The model is loaded via the normal startup
event, which relies on model/current_version.json + model_v{N}.pkl having
been created by `python model/train.py` (this is done automatically in CI
before tests run, and also at Docker build time).
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))

from serving.app import app

client = TestClient(app)
client.__enter__()  # trigger startup event (loads model)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_expected_structure():
    payload = {
        "features": {
            "feature_1": 0.5,
            "feature_2": 5.0,
            "feature_3": 3.0,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "model_version" in data
    assert isinstance(data["prediction"], int)
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_endpoint_missing_feature_returns_400():
    payload = {"features": {"feature_1": 0.5}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
