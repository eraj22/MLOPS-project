"""
drift_detector.py
------------------
Computes simple per-feature statistics (mean, std) for an incoming batch of
records and compares them to a stored baseline to detect distribution drift.

Drift is flagged for a feature if the relative change in mean exceeds
DRIFT_THRESHOLD (configurable). This is intentionally simple (per the
project spec, "simple statistics") rather than a full statistical test
(e.g. KS-test/PSI), but the structure makes it easy to swap in a library
like Evidently AI later.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.20"))


class DriftDetector:
    def __init__(self, baseline_path: str = "data/baseline_stats.json",
                 threshold: float = DEFAULT_THRESHOLD):
        self.baseline_path = Path(baseline_path)
        self.threshold = threshold
        self.baseline: Dict[str, Dict[str, float]] = {}
        self._load_baseline()

    def _load_baseline(self) -> None:
        if self.baseline_path.exists():
            with open(self.baseline_path, "r") as f:
                self.baseline = json.load(f)

    def save_baseline(self) -> None:
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, "w") as f:
            json.dump(self.baseline, f, indent=2)

    def compute_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute mean/std for every numeric column in the batch."""
        stats = {}
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) == 0:
                continue
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
            }
        return stats

    def set_baseline_if_missing(self, df: pd.DataFrame) -> None:
        """If no baseline exists yet, establish one from the first batch."""
        if not self.baseline:
            self.baseline = self.compute_stats(df)
            self.save_baseline()

    def check_drift(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Compare current batch statistics to the baseline.

        Returns:
            (drift_detected: bool, drifted_features: list[str])
        """
        current_stats = self.compute_stats(df)
        drifted_features = []

        for feature, base in self.baseline.items():
            if feature not in current_stats:
                continue  # handled separately by schema-change logic
            curr = current_stats[feature]
            base_mean = base.get("mean", 0.0)

            if base_mean == 0:
                # Avoid division by zero: use absolute difference instead
                relative_change = abs(curr["mean"] - base_mean)
            else:
                relative_change = abs(curr["mean"] - base_mean) / abs(base_mean)

            if relative_change > self.threshold:
                drifted_features.append(feature)

        drift_detected = len(drifted_features) > 0
        return drift_detected, drifted_features

    def update_baseline(self, df: pd.DataFrame) -> None:
        """Refresh baseline stats (e.g. after retraining on new data)."""
        self.baseline = self.compute_stats(df)
        self.save_baseline()
