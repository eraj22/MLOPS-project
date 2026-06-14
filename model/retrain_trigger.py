"""
model/retrain_trigger.py
--------------------------
Part 2 deliverable: orchestrates automated retraining.

Checks for any of the three retraining conditions:
    1. Current model accuracy (from exporter / current_version.json) < TARGET_ACCURACY
    2. Distribution drift flag set by ingestion.py (data/retrain_needed.flag)
    3. Schema change flag set by ingestion.py (data/retrain_needed.flag)

If triggered:
    - Increments retrain_count_total Prometheus counter.
    - Logs the reason.
    - Calls model/train.py to retrain.
    - Sends a Slack notification with the reason and new accuracy.
    - (Redeployment to AWS is handled by deploy/deploy.sh, normally invoked
      by CI/CD after a successful retrain - see README.)

Usage:
    python model/retrain_trigger.py            # check once and act if needed
    python model/retrain_trigger.py --force    # force retraining regardless of flags
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.train import train, CURRENT_VERSION_PATH, TARGET_ACCURACY
from ingestion.slack_alerts import send_slack_message
from exporter.metrics import RETRAIN_COUNT_TOTAL, MODEL_ACCURACY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("retrain_trigger")

BASE_DIR = Path(__file__).resolve().parent.parent
RETRAIN_FLAG_PATH = BASE_DIR / "data" / "retrain_needed.flag"
RECORD_COUNTER_PATH = BASE_DIR / "data" / "new_record_counter.txt"


def _current_accuracy() -> float | None:
    if CURRENT_VERSION_PATH.exists():
        with open(CURRENT_VERSION_PATH, "r") as f:
            data = json.load(f)
        return data.get("accuracy")
    return None


def _check_conditions() -> tuple[bool, str]:
    """Return (should_retrain, reason)."""
    # Condition 1: accuracy below threshold
    accuracy = _current_accuracy()
    if accuracy is not None and accuracy < TARGET_ACCURACY:
        return True, f"accuracy_below_threshold current={accuracy} target={TARGET_ACCURACY}"

    # Conditions 2 & 3: flag file written by ingestion.py (drift / schema change /
    # record-count threshold)
    if RETRAIN_FLAG_PATH.exists():
        with open(RETRAIN_FLAG_PATH, "r") as f:
            flag_data = json.load(f)
        return True, flag_data.get("reason", "flagged_by_ingestion")

    return False, "no_trigger_condition_met"


def _clear_flags() -> None:
    if RETRAIN_FLAG_PATH.exists():
        RETRAIN_FLAG_PATH.unlink()
    if RECORD_COUNTER_PATH.exists():
        RECORD_COUNTER_PATH.write_text("0")


def run(force: bool = False) -> dict | None:
    should_retrain, reason = (True, "forced") if force else _check_conditions()

    if not should_retrain:
        logger.info("No retraining needed: %s", reason)
        return None

    logger.info("Retraining triggered. Reason: %s", reason)
    RETRAIN_COUNT_TOTAL.inc()

    metadata = train()
    new_accuracy = metadata["accuracy"]
    MODEL_ACCURACY.set(new_accuracy)

    send_slack_message(
        f":arrows_counterclockwise: Model retraining triggered.\n"
        f"*Reason:* {reason}\n"
        f"*New version:* model_v{metadata['version']}\n"
        f"*New accuracy:* {new_accuracy}"
    )

    _clear_flags()
    logger.info("Retraining complete. New accuracy=%.4f (version=%d)", new_accuracy, metadata["version"])

    # NOTE: Redeployment to AWS EC2 is performed by deploy/deploy.sh.
    # In CI/CD this script is followed by a call to that deploy script.
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-retraining orchestrator")
    parser.add_argument("--force", action="store_true", help="Force retraining regardless of conditions")
    args = parser.parse_args()

    run(force=args.force)
