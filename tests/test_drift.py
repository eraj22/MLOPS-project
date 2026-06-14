"""
tests/test_drift.py
----------------------
Unit test for distribution drift detection (Part 1 / Section 9.3.2).

Provides a reference distribution and a clearly shifted distribution and
asserts that drift is flagged.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.drift_detector import DriftDetector


def test_drift_detected_on_shifted_distribution(tmp_path):
    baseline_path = tmp_path / "baseline_stats.json"
    detector = DriftDetector(baseline_path=str(baseline_path), threshold=0.20)

    # Reference distribution: feature centered around 10
    reference_df = pd.DataFrame({"feature_x": [10.0, 10.1, 9.9, 10.2, 9.8]})
    detector.set_baseline_if_missing(reference_df)

    # Clearly shifted distribution: feature centered around 20 (>100% change)
    shifted_df = pd.DataFrame({"feature_x": [20.0, 20.1, 19.9, 20.2, 19.8]})
    drift_detected, drifted_features = detector.check_drift(shifted_df)

    assert drift_detected is True
    assert "feature_x" in drifted_features


def test_no_drift_on_similar_distribution(tmp_path):
    baseline_path = tmp_path / "baseline_stats.json"
    detector = DriftDetector(baseline_path=str(baseline_path), threshold=0.20)

    reference_df = pd.DataFrame({"feature_x": [10.0, 10.1, 9.9, 10.2, 9.8]})
    detector.set_baseline_if_missing(reference_df)

    similar_df = pd.DataFrame({"feature_x": [10.05, 10.0, 9.95, 10.1, 9.9]})
    drift_detected, drifted_features = detector.check_drift(similar_df)

    assert drift_detected is False
    assert drifted_features == []
