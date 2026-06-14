#!/bin/sh
# Alertmanager's config loader does NOT expand ${ENV_VARS} natively.
# This entrypoint substitutes SLACK_WEBHOOK_URL into the config template
# before starting Alertmanager. Used via docker-compose entrypoint override.

set -e

envsubst < /etc/alertmanager/alertmanager.yml.template > /etc/alertmanager/alertmanager.yml

exec /bin/alertmanager --config.file=/etc/alertmanager/alertmanager.yml "$@"
