"""
serving/app.py
----------------
Part 3 & Part 4 deliverable: FastAPI ML inference service exposing:
    - POST /predict  -> prediction + confidence
    - GET  /metrics  -> Prometheus-format metrics
    - GET  /health   -> {"status": "ok"}

Run locally:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000

The model is loaded from model/current_version.json + the referenced
model_v{N}.pkl. On startup, MODEL_ACCURACY is set from this metadata so the
gauge reflects the deployed model's validation accuracy immediately.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

sys.path.append(str(Path(__file__).resolve().parent.parent))

from exporter.metrics import (
    MODEL_ACCURACY,
    RESPONSE_DELAY_SECONDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("serving")

BASE_DIR = Path(__file__).resolve().parent.parent
CURRENT_VERSION_PATH = BASE_DIR / "model" / "current_version.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="MLOps Inference API", lifespan=lifespan)

_model_bundle = None
_feature_columns: list[str] = []
_model_version: int = 0


class PredictRequest(BaseModel):
    features: Dict[str, float]


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: int


def _load_model() -> None:
    global _model_bundle, _feature_columns, _model_version

    if not CURRENT_VERSION_PATH.exists():
        raise RuntimeError(
            f"{CURRENT_VERSION_PATH} not found. Run `python model/train.py` first."
        )

    with open(CURRENT_VERSION_PATH, "r") as f:
        meta = json.load(f)

    model_path = BASE_DIR / meta["model_path"]
    bundle = joblib.load(model_path)

    _model_bundle = bundle["model"]
    _feature_columns = bundle["feature_columns"]
    _model_version = meta["version"]

    MODEL_ACCURACY.set(meta["accuracy"])
    logger.info(
        "Loaded model_v%d from %s (accuracy=%.4f, features=%s)",
        _model_version, model_path, meta["accuracy"], _feature_columns,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        missing = [c for c in _feature_columns if c not in request.features]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing}. Expected: {_feature_columns}",
            )

        ordered = pd.DataFrame(
            [[request.features[col] for col in _feature_columns]],
            columns=_feature_columns,
        )

        prediction = _model_bundle.predict(ordered)[0]
        proba = _model_bundle.predict_proba(ordered)[0]
        confidence = float(max(proba))

        return PredictResponse(
            prediction=int(prediction),
            confidence=confidence,
            model_version=_model_version,
        )
    finally:
        RESPONSE_DELAY_SECONDS.observe(time.time() - start)


@app.post("/reload")
def reload_model() -> dict:
    """Hot-reload the model after retraining without restarting the container."""
    _load_model()
    return {"status": "reloaded", "model_version": _model_version}


@app.post("/debug/trigger/{condition}")
def debug_trigger(condition: str) -> dict:
    """
    DEMO/TEST ONLY: manually flips metric values so you can capture
    screenshots of each Slack alert firing (Part 6 deliverable).

    Valid `condition` values:
        datalake, feature_added, feature_removed, drift, clear_drift,
        latency, low_accuracy, restore_accuracy
    """
    from exporter.metrics import (
        DATALAKE_UNAVAILABLE,
        FEATURE_ADDED,
        FEATURE_REMOVED,
        DISTRIBUTION_DRIFT_DETECTED,
        RESPONSE_DELAY_SECONDS,
        MODEL_ACCURACY,
    )

    actions = {
        "datalake": lambda: DATALAKE_UNAVAILABLE.inc(),
        "feature_added": lambda: FEATURE_ADDED.inc(),
        "feature_removed": lambda: FEATURE_REMOVED.inc(),
        "drift": lambda: DISTRIBUTION_DRIFT_DETECTED.set(1),
        "clear_drift": lambda: DISTRIBUTION_DRIFT_DETECTED.set(0),
        "latency": lambda: [RESPONSE_DELAY_SECONDS.observe(1.5) for _ in range(20)],
        "low_accuracy": lambda: MODEL_ACCURACY.set(0.5),
        "restore_accuracy": lambda: MODEL_ACCURACY.set(0.95),
    }

    if condition not in actions:
        raise HTTPException(status_code=400, detail=f"Unknown condition. Valid: {list(actions.keys())}")

    actions[condition]()
    return {"status": "triggered", "condition": condition}
