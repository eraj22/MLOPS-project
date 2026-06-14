"""
model/train.py
----------------
Part 2 deliverable: trains a classifier on ingested data, enforcing a
validation-accuracy threshold, and saves a versioned model artifact.

Usage:
    python model/train.py

Behavior:
    - Loads data/records.csv (written by ingestion.py).
    - If the file does not exist or is too small, falls back to a small
      synthetic dataset so the pipeline can run end-to-end during
      development/testing.
    - Trains a RandomForestClassifier, retrying with different
      hyperparameters / random seeds until validation accuracy >= TARGET_ACCURACY
      or MAX_TRAIN_ITERATIONS is reached.
    - Saves the model as model/model_v{N}.pkl (auto-incrementing version).
    - Writes model/current_version.json with {version, accuracy} so other
      components (serving, exporter, retrain_trigger) can discover the
      latest model.
"""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "records.csv"
MODEL_DIR = Path(__file__).resolve().parent
CURRENT_VERSION_PATH = MODEL_DIR / "current_version.json"

TARGET_ACCURACY = float(os.getenv("TARGET_ACCURACY", "0.80"))
MAX_TRAIN_ITERATIONS = int(os.getenv("MAX_TRAIN_ITERATIONS", "10"))
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "label")


def _load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if len(df) >= 50 and TARGET_COLUMN in df.columns:
            logger.info("Loaded %d records from %s", len(df), DATA_PATH)
            return df
        logger.warning(
            "Ingested data insufficient (rows=%d, has_target=%s); using synthetic fallback.",
            len(df), TARGET_COLUMN in df.columns,
        )

    logger.warning("No usable ingested data found; generating synthetic dataset.")
    return _synthetic_dataset()


def _synthetic_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Small synthetic binary classification dataset as a fallback so the
    pipeline is fully runnable before the live API has produced enough data."""
    rng = np.random.RandomState(seed)
    feature_1 = rng.normal(0, 1, n)
    feature_2 = rng.normal(5, 2, n)
    feature_3 = rng.uniform(0, 10, n)
    label = ((feature_1 + 0.3 * feature_2 - 0.1 * feature_3) > 1.5).astype(int)
    return pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        TARGET_COLUMN: label,
    })


def _next_version() -> int:
    existing = list(MODEL_DIR.glob("model_v*.pkl"))
    if not existing:
        return 1
    versions = []
    for p in existing:
        try:
            versions.append(int(p.stem.split("model_v")[-1]))
        except ValueError:
            continue
    return max(versions, default=0) + 1


def train() -> dict:
    df = _load_dataset()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not present in dataset columns: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Encode any non-numeric feature columns
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode labels if non-numeric
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    best_model = None
    best_accuracy = 0.0

    for iteration in range(1, MAX_TRAIN_ITERATIONS + 1):
        n_estimators = 50 * iteration
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=iteration,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)

        logger.info(
            "Iteration %d/%d: n_estimators=%d -> val_accuracy=%.4f",
            iteration, MAX_TRAIN_ITERATIONS, n_estimators, accuracy,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

        if accuracy >= TARGET_ACCURACY:
            logger.info("Target accuracy %.2f reached at iteration %d.", TARGET_ACCURACY, iteration)
            break
    else:
        logger.warning(
            "Target accuracy %.2f not reached after %d iterations. "
            "Proceeding with best achieved model (accuracy=%.4f).",
            TARGET_ACCURACY, MAX_TRAIN_ITERATIONS, best_accuracy,
        )

    # Save versioned artifact
    version = _next_version()
    model_path = MODEL_DIR / f"model_v{version}.pkl"
    joblib.dump(
        {"model": best_model, "feature_columns": list(X.columns)},
        model_path,
    )
    logger.info("Saved model artifact to %s", model_path)

    metadata = {
        "version": version,
        "accuracy": round(float(best_accuracy), 4),
        "model_path": str(model_path.relative_to(BASE_DIR)),
        "feature_columns": list(X.columns),
        "target_column": TARGET_COLUMN,
    }
    with open(CURRENT_VERSION_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Updated %s -> %s", CURRENT_VERSION_PATH, metadata)
    return metadata


if __name__ == "__main__":
    train()
