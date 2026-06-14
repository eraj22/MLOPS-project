"""
exporter/metrics.py
--------------------
Central definitions for all Prometheus metrics required by Part 4.

Importing this module gives every other component (ingestion, training,
serving) access to the SAME metric objects, so they all update a shared
in-process registry. The FastAPI inference service (serving/app.py) exposes
these at /metrics using prometheus_client's generate_latest().
"""

from prometheus_client import Counter, Gauge, Histogram

# 1. Current validation accuracy of the deployed model (0.0 - 1.0)
MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current validation accuracy of the deployed model",
)

# 2. Total number of records ingested from the API since startup
RECORDS_PROCESSED_TOTAL = Counter(
    "records_processed_total",
    "Total number of records ingested from the /records API since startup",
)

# 3. Total number of times the model has been retrained
RETRAIN_COUNT_TOTAL = Counter(
    "retrain_count_total",
    "Total number of times the model has been retrained",
)

# 4. Set to 1 when drift is detected in the current batch, 0 otherwise
DISTRIBUTION_DRIFT_DETECTED = Gauge(
    "distribution_drift_detected",
    "1 if distribution drift was detected in the most recent batch, else 0",
)

# 5. Number of features added to the schema since startup
# NOTE: Gauge (not Counter) is used here so the exposed metric name is
# exactly "feature_added" as required by the spec / alert rules.
# prometheus_client automatically appends "_total" to Counter names that
# don't already end in _total. .inc() works the same on a Gauge.
FEATURE_ADDED = Gauge(
    "feature_added",
    "Number of features added to the schema since startup",
)

# 6. Number of features removed from the schema since startup
FEATURE_REMOVED = Gauge(
    "feature_removed",
    "Number of features removed from the schema since startup",
)

# 7. Number of times the /records endpoint returned 503
DATALAKE_UNAVAILABLE = Gauge(
    "datalake_unavailable",
    "Number of times the /records endpoint returned HTTP 503",
)

# 8. Latency of each /predict API call in seconds
RESPONSE_DELAY_SECONDS = Histogram(
    "response_delay_seconds",
    "Latency of each /predict API call in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)


def set_model_metadata(version: int, accuracy: float) -> None:
    """Convenience helper: update accuracy gauge and log the model version."""
    MODEL_ACCURACY.set(accuracy)
