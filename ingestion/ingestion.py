"""
ingestion/ingestion.py
-----------------------
Part 1 deliverable: polls the provided /records HTTP endpoint, stores data
locally, detects schema changes and distribution drift, handles 503s, and
triggers retraining when needed.

Usage:
    python ingestion/ingestion.py            # run continuously (loop)
    python ingestion/ingestion.py --once      # run a single iteration

Environment variables (see .env.example):
    RECORDS_API_URL, POLL_INTERVAL_SECONDS, DRIFT_THRESHOLD,
    RETRAIN_RECORD_THRESHOLD, SLACK_WEBHOOK_URL
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Allow running this file directly (python ingestion/ingestion.py) as well
# as via `python -m ingestion.ingestion`.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.drift_detector import DriftDetector
from ingestion.slack_alerts import send_slack_message
from exporter.metrics import (
    RECORDS_PROCESSED_TOTAL,
    FEATURE_ADDED,
    FEATURE_REMOVED,
    DATALAKE_UNAVAILABLE,
    DISTRIBUTION_DRIFT_DETECTED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion")

RECORDS_API_URL = os.getenv("RECORDS_API_URL", "http://149.40.228.124:6500/records")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
RETRAIN_RECORD_THRESHOLD = int(os.getenv("RETRAIN_RECORD_THRESHOLD", "200"))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_PATH = DATA_DIR / "records.csv"
SCHEMA_STATE_PATH = DATA_DIR / "last_schema.json"
RETRAIN_FLAG_PATH = DATA_DIR / "retrain_needed.flag"


def load_last_schema() -> list:
    if SCHEMA_STATE_PATH.exists():
        with open(SCHEMA_STATE_PATH, "r") as f:
            return json.load(f)
    return []


def save_last_schema(schema: list) -> None:
    with open(SCHEMA_STATE_PATH, "w") as f:
        json.dump(schema, f)


def append_records(records: list) -> None:
    """Append a batch of records to the local CSV store."""
    if not records:
        return
    df = pd.DataFrame(records)
    if RAW_DATA_PATH.exists():
        df.to_csv(RAW_DATA_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(RAW_DATA_PATH, mode="w", header=True, index=False)


def count_new_records_since_last_train() -> int:
    """Counts rows accumulated since the last retraining checkpoint."""
    counter_path = DATA_DIR / "new_record_counter.txt"
    if counter_path.exists():
        return int(counter_path.read_text().strip() or 0)
    return 0


def increment_new_record_counter(n: int) -> int:
    counter_path = DATA_DIR / "new_record_counter.txt"
    current = count_new_records_since_last_train()
    new_total = current + n
    counter_path.write_text(str(new_total))
    return new_total


def reset_new_record_counter() -> None:
    counter_path = DATA_DIR / "new_record_counter.txt"
    counter_path.write_text("0")


def signal_retrain(reason: str) -> None:
    """Write a flag file the retrain_trigger.py / scheduler can pick up."""
    with open(RETRAIN_FLAG_PATH, "w") as f:
        json.dump({"reason": reason, "timestamp": time.time()}, f)
    logger.info("Retraining signal raised: %s", reason)


def fetch_records():
    """Call the /records endpoint. Returns parsed JSON (list or dict) or None on 503/error."""
    try:
        resp = requests.get(RECORDS_API_URL, timeout=15)
    except requests.RequestException as exc:
        logger.error("Request to %s failed: %s", RECORDS_API_URL, exc)
        return None

    if resp.status_code == 503:
        logger.warning("Data source unavailable (503).")
        DATALAKE_UNAVAILABLE.inc()
        send_slack_message(
            ":warning: Data source returned 503. Check API availability."
        )
        return None

    if resp.status_code != 200:
        logger.error("Unexpected status code %s from %s", resp.status_code, RECORDS_API_URL)
        return None

    try:
        return resp.json()
    except ValueError:
        logger.error("Failed to parse JSON response from %s", RECORDS_API_URL)
        return None


def flatten_record(record: dict) -> dict:
    """
    The live /records API returns rows shaped like:
        {"features": [1.30, 4.42], "label": 1}

    This flattens that into a flat dict so it can become DataFrame columns:
        {"feature_0": 1.30, "feature_1": 4.42, "label": 1}

    Any other top-level keys (besides "features") are passed through as-is,
    which also lets us detect genuinely new/removed top-level fields if the
    API evolves.
    """
    flat = {}
    for key, value in record.items():
        if key == "features" and isinstance(value, list):
            for i, v in enumerate(value):
                flat[f"feature_{i}"] = v
        else:
            flat[key] = value
    return flat


def process_batch(payload, drift_detector: DriftDetector) -> None:
    # The spec describes a {"schema": [...], "records": [...]} envelope, but
    # the live API returns a flat JSON list of records directly:
    #   [{"features": [f0, f1, ...], "label": 0/1}, ...]
    # Support both shapes.
    if isinstance(payload, dict):
        records = payload.get("records", [])
        explicit_schema = payload.get("schema")
    else:
        records = payload
        explicit_schema = None

    if not records:
        logger.info("Received empty batch; nothing to process.")
        return

    flat_records = [flatten_record(r) for r in records]
    df = pd.DataFrame(flat_records)

    # Derive schema from the flattened columns (e.g. feature_0, feature_1, label)
    # unless the API explicitly provided one.
    schema = explicit_schema if explicit_schema else list(df.columns)

    # --- 1. Schema change detection -----------------------------------
    last_schema = load_last_schema()
    if last_schema:
        added = set(schema) - set(last_schema)
        removed = set(last_schema) - set(schema)

        for feature in added:
            logger.info("Schema change: feature added -> %s", feature)
            FEATURE_ADDED.inc()
            send_slack_message(
                f":heavy_plus_sign: New feature detected in schema: `{feature}`. "
                f"Retraining may be required."
            )

        for feature in removed:
            logger.info("Schema change: feature removed -> %s", feature)
            FEATURE_REMOVED.inc()
            send_slack_message(
                f":heavy_minus_sign: Feature dropped from schema: `{feature}`. "
                f"Verify pipeline compatibility."
            )

        if added or removed:
            signal_retrain(f"schema_change added={list(added)} removed={list(removed)}")
    else:
        logger.info("No previous schema found; establishing baseline schema: %s", schema)

    save_last_schema(schema)

    # --- 2. Store the batch locally ------------------------------------
    append_records(flat_records)
    RECORDS_PROCESSED_TOTAL.inc(len(flat_records))
    new_total = increment_new_record_counter(len(flat_records))
    logger.info("Stored %d records (total since last retrain: %d).", len(flat_records), new_total)

    # --- 3. Distribution drift detection --------------------------------
    drift_detector.set_baseline_if_missing(df)
    drift_detected, drifted_features = drift_detector.check_drift(df)
    DISTRIBUTION_DRIFT_DETECTED.set(1 if drift_detected else 0)

    if drift_detected:
        logger.warning("Distribution drift detected in features: %s", drifted_features)
        send_slack_message(
            f":chart_with_downwards_trend: Data distribution drift detected in "
            f"features: {', '.join(drifted_features)}. Model may be stale."
        )
        signal_retrain(f"distribution_drift features={drifted_features}")

    # --- 4. Trigger retraining if enough new data accumulated -----------
    if new_total >= RETRAIN_RECORD_THRESHOLD:
        signal_retrain(f"record_threshold_reached count={new_total}")


def run_once() -> None:
    drift_detector = DriftDetector()
    payload = fetch_records()
    if payload is not None:
        process_batch(payload, drift_detector)


def run_loop() -> None:
    drift_detector = DriftDetector()
    logger.info(
        "Starting ingestion loop. Polling %s every %ds.",
        RECORDS_API_URL,
        POLL_INTERVAL_SECONDS,
    )
    while True:
        payload = fetch_records()
        if payload is not None:
            process_batch(payload, drift_detector)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data ingestion & schema monitor")
    parser.add_argument("--once", action="store_true", help="Run a single fetch/process cycle")
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_loop()
