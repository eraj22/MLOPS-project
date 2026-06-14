"""
tests/test_schema.py
----------------------
Unit test for schema change detection (Part 1 / Section 9.3.1).

Simulates two batches with differing schemas and asserts that the correct
feature_added / feature_removed counters are incremented.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion import ingestion
from exporter.metrics import FEATURE_ADDED, FEATURE_REMOVED


def test_schema_change_detection(tmp_path, monkeypatch):
    # Redirect data dir + schema state file to a temp location
    monkeypatch.setattr(ingestion, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ingestion, "SCHEMA_STATE_PATH", tmp_path / "last_schema.json")
    monkeypatch.setattr(ingestion, "RAW_DATA_PATH", tmp_path / "records.csv")
    monkeypatch.setattr(ingestion, "RETRAIN_FLAG_PATH", tmp_path / "retrain_needed.flag")

    before_added = FEATURE_ADDED._value.get()
    before_removed = FEATURE_REMOVED._value.get()

    # First batch establishes the baseline schema (feature_a, feature_b)
    batch_1 = {
        "schema": ["feature_a", "feature_b"],
        "records": [{"feature_a": 1.0, "feature_b": 2.0}],
    }

    drift_detector = _stub_drift_detector()
    ingestion.process_batch(batch_1, drift_detector)

    # Second batch: feature_b removed, feature_c added
    batch_2 = {
        "schema": ["feature_a", "feature_c"],
        "records": [{"feature_a": 1.5, "feature_c": 9.0}],
    }
    ingestion.process_batch(batch_2, drift_detector)

    after_added = FEATURE_ADDED._value.get()
    after_removed = FEATURE_REMOVED._value.get()

    assert after_added == before_added + 1, "feature_added counter should increment by 1"
    assert after_removed == before_removed + 1, "feature_removed counter should increment by 1"

    # The schema state file should now reflect the latest schema
    with open(tmp_path / "last_schema.json") as f:
        saved_schema = json.load(f)
    assert saved_schema == ["feature_a", "feature_c"]


def test_schema_change_detection_flat_list_format(tmp_path, monkeypatch):
    """The live /records API returns a flat JSON list of
    {"features": [...], "label": ...} rows (no schema/records envelope).
    This test mirrors that real shape and checks that adding an extra
    feature column (e.g. features array growing from 2 -> 3 elements)
    is detected as feature_added."""
    monkeypatch.setattr(ingestion, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ingestion, "SCHEMA_STATE_PATH", tmp_path / "last_schema.json")
    monkeypatch.setattr(ingestion, "RAW_DATA_PATH", tmp_path / "records.csv")
    monkeypatch.setattr(ingestion, "RETRAIN_FLAG_PATH", tmp_path / "retrain_needed.flag")

    before_added = FEATURE_ADDED._value.get()

    drift_detector = _stub_drift_detector()

    # Batch 1: 2 features -> schema = [feature_0, feature_1, label]
    batch_1 = [
        {"features": [1.3, 4.4], "label": 1},
        {"features": [0.05, -0.18], "label": 0},
    ]
    ingestion.process_batch(batch_1, drift_detector)

    # Batch 2: 3 features -> schema = [feature_0, feature_1, feature_2, label]
    batch_2 = [
        {"features": [1.1, 4.0, 0.9], "label": 1},
        {"features": [0.1, -0.2, 0.3], "label": 0},
    ]
    ingestion.process_batch(batch_2, drift_detector)

    after_added = FEATURE_ADDED._value.get()
    assert after_added == before_added + 1, "feature_2 should be detected as feature_added"

    with open(tmp_path / "last_schema.json") as f:
        saved_schema = json.load(f)
    assert "feature_2" in saved_schema


def _stub_drift_detector():
    """Drift detector pointed at a throwaway baseline file so tests don't
    touch the real data/baseline_stats.json."""
    from ingestion.drift_detector import DriftDetector
    tmpdir = tempfile.mkdtemp()
    return DriftDetector(baseline_path=str(Path(tmpdir) / "baseline_stats.json"))
