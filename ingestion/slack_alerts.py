"""
slack_alerts.py
----------------
Small shared helper to send ad-hoc Slack notifications directly from Python
code (in addition to the Prometheus Alertmanager -> Slack route).

This is used by:
 - ingestion.py (schema change / drift / 503 alerts)
 - retrain_trigger.py (retraining notifications)

The webhook URL is read from the SLACK_WEBHOOK_URL environment variable and
is never hardcoded.
"""

import os
import logging

import requests

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def send_slack_message(text: str) -> bool:
    """Send a simple text message to the configured Slack webhook.

    Returns True on success, False otherwise. Never raises - alerting must
    not crash the main pipeline.
    """
    if not SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL not set; skipping Slack alert: %s", text)
        return False

    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
        if resp.status_code != 200:
            logger.error("Slack webhook returned %s: %s", resp.status_code, resp.text)
            return False
        return True
    except requests.RequestException as exc:
        logger.error("Failed to send Slack alert: %s", exc)
        return False
