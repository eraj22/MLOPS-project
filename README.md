# MLOps Project — End-to-End Pipeline with Monitoring, Alerting & CI/CD

**Course:** Machine Learning Operations (MLOps), Spring 2026 — FAST NUCES
**Team member(s):22i1296 ERAJ ZAMAN 

## 1. Project Description

This repository implements a complete, production-style MLOps pipeline:

1. **Ingestion** (`ingestion/`) — polls a live `/records` API, stores data, detects schema
   changes and distribution drift, handles `503` outages, and signals retraining.
2. **Training & Auto-Retraining** (`model/`) — trains a `RandomForestClassifier` until
   validation accuracy ≥ 0.80, versions model artifacts, and retrains automatically
   when accuracy drops, drift is detected, or the schema changes.
3. **Serving** (`serving/`) — a FastAPI inference service exposing `/predict`,
   `/metrics`, and `/health`, containerized with Docker and deployable to AWS EC2.
4. **Metrics** (`exporter/`) — defines all 8 required Prometheus metrics.
5. **Observability stack** (`docker-compose.yml`, `prometheus/`, `grafana/`,
   `alertmanager/`) — Prometheus + Grafana + Alertmanager running via Docker Compose.
6. **Alerting** — 7 Prometheus alert rules routed to Slack via Alertmanager.
7. **CI/CD** (`.github/workflows/mlops-ci.yml`) — lint, test, build/push Docker image,
   and deploy to EC2 on every push to `main`.

---

## 2. Repository Structure

```
mlops-project/
├── .github/workflows/mlops-ci.yml
├── ingestion/
│   ├── ingestion.py
│   ├── drift_detector.py
│   └── slack_alerts.py
├── model/
│   ├── train.py
│   ├── retrain_trigger.py
│   └── model_v1.pkl            (generated)
├── serving/
│   └── app.py
├── exporter/
│   └── metrics.py
├── prometheus/
│   ├── prometheus.yml
│   └── alert_rules.yml
├── alertmanager/
│   ├── alertmanager.yml.template
│   └── entrypoint.sh
├── grafana/dashboards/mlops_dashboard.json
├── deploy/deploy.sh
├── tests/
│   ├── test_schema.py
│   ├── test_drift.py
│   └── test_predict.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 3. Prerequisites

- Python 3.10+
- Docker & Docker Compose
- An AWS account (free-tier EC2 instance)
- A Docker Hub account
- A Slack workspace with an Incoming Webhook configured
  (Slack → Apps → Incoming Webhooks → Add to Workspace → copy the webhook URL)

---

## 4. Local Setup (Run Everything on Your Machine First)

### 4.1 Clone & install dependencies

```bash
git clone <your-repo-url>
cd mlops-project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Configure environment variables

```bash
cp .env.example .env
# Edit .env and set SLACK_WEBHOOK_URL to your real Slack webhook
export $(grep -v '^#' .env | xargs)   # load vars into current shell (Linux/Mac)
```

### 4.3 Run ingestion once (fetches data, detects schema/drift)

```bash
python ingestion/ingestion.py --once
```

This calls `http://149.40.228.124:6500/records`, stores results to
`data/records.csv`, updates `data/last_schema.json`, and computes drift
statistics in `data/baseline_stats.json`. If the endpoint returns 503, the
script logs it, increments `datalake_unavailable`, and sends a Slack alert.

To run continuously (every `POLL_INTERVAL_SECONDS`):

```bash
python ingestion/ingestion.py
```

### 4.4 Train the model

```bash
python model/train.py
```

This produces `model/model_v1.pkl` (or the next version number) and
`model/current_version.json`. If `data/records.csv` doesn't yet have enough
rows (≥ 50) or a `label` column, a synthetic dataset is used so the pipeline
remains fully runnable end-to-end.

> **Note:** If your live API's records don't include an explicit label
> column, set `TARGET_COLUMN` in `.env` to whatever target field the API
> provides, or adapt `model/train.py`'s feature engineering as needed for
> your specific schema.

### 4.5 Run the inference API locally

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Test it:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 0.5, "feature_2": 5.0, "feature_3": 3.0}}'
# {"prediction":1,"confidence":0.64,"model_version":1}

curl http://localhost:8000/metrics | head -30
```

### 4.6 Run the auto-retraining check

```bash
python model/retrain_trigger.py          # checks conditions, retrains if needed
python model/retrain_trigger.py --force   # force a retrain regardless
```

### 4.7 Run unit tests & lint

```bash
flake8 ingestion model serving exporter tests
pytest tests/ -v
```

All 6 tests (schema detection ×1, drift detection ×2, predict endpoint ×3)
should pass.

---

## 5. AWS EC2 Deployment (Part 3)

### 5.1 Launch the EC2 instance

1. AWS Console → EC2 → Launch Instance.
2. AMI: **Ubuntu 22.04 LTS**, type: **t2.micro** / **t3.micro** (free tier).
3. Create/select a key pair (download the `.pem` file — needed for SSH and
   for the `EC2_SSH_KEY` GitHub secret).
4. **Security Group** — allow inbound TCP on:
   - `22` (SSH)
   - `80` (HTTP — inference API via deploy script mapping)
   - `8000` (FastAPI direct)
   - `9090` (Prometheus, if running stack on same host)
   - `3000` (Grafana, if running stack on same host)
   - `9093` (Alertmanager, if running stack on same host)
5. Launch, then note the **public IPv4 address** — this is your `EC2_HOST`.

### 5.2 Install Docker on the instance

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu
# log out and back in for group change to apply
```

### 5.3 Build & push the image (from your local machine)

```bash
export DOCKER_USERNAME=<your-dockerhub-username>
export DOCKER_PASSWORD=<your-dockerhub-token>
export DOCKER_IMAGE_NAME=$DOCKER_USERNAME/mlops-inference
export IMAGE_TAG=latest

./deploy/deploy.sh build-and-push
```

### 5.4 Deploy to EC2

```bash
export EC2_HOST=<EC2_PUBLIC_IP>
export EC2_USER=ubuntu
export EC2_SSH_KEY_PATH=./your-key.pem
export SLACK_WEBHOOK_URL=<your-slack-webhook>

./deploy/deploy.sh remote-deploy
```

Or do both steps at once: `./deploy/deploy.sh all`

### 5.5 Verify

```bash
curl http://<EC2_PUBLIC_IP>:8000/health
curl http://<EC2_PUBLIC_IP>:8000/metrics
curl -X POST http://<EC2_PUBLIC_IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 0.5, "feature_2": 5.0, "feature_3": 3.0}}'
```

> **EC2 Public IP used for this project:** `<FILL_IN_YOUR_EC2_IP>`

---

## 6. Observability Stack — Prometheus, Grafana, Alertmanager (Part 5)

### 6.1 Point Prometheus at your EC2 instance

Edit `prometheus/prometheus.yml` and replace `YOUR_EC2_PUBLIC_IP` with your
actual EC2 public IP (job `mlops_inference_ec2`).

### 6.2 Start the stack (run on your local machine or a separate EC2 host)

```bash
export SLACK_WEBHOOK_URL=<your-slack-webhook>
docker compose up -d
```

This starts:
- **Prometheus** → http://localhost:9090
- **Grafana** → http://localhost:3000 (login: `admin` / `admin`)
- **Alertmanager** → http://localhost:9093
- (optional) a local copy of the inference service → http://localhost:8000

### 6.3 Verify Prometheus targets

Go to http://localhost:9090/targets — both `mlops_inference_ec2` and
`mlops_inference_local` (if enabled) should show as `UP`.

### 6.4 Import the Grafana dashboard

1. Open Grafana → http://localhost:3000 (admin/admin).
2. Add a Prometheus data source: Connections → Data sources → Add data
   source → Prometheus → URL `http://prometheus:9090` → Save & Test.
3. Dashboards → New → Import → upload
   `grafana/dashboards/mlops_dashboard.json`.
4. When prompted, select your Prometheus data source for the
   `DS_PROMETHEUS` variable.

The dashboard includes panels for: model accuracy over time, records
processed rate, retrain count, distribution drift indicator, datalake
unavailable count, feature added/removed, and P95 response latency.

---

## 7. Slack Alerts (Part 6)

### 7.1 Create a Slack Incoming Webhook

1. Go to https://api.slack.com/apps → Create New App → From scratch.
2. Add the **Incoming Webhooks** feature, activate it.
3. Click "Add New Webhook to Workspace", choose `#mlops-alerts` (create the
   channel first if needed).
4. Copy the webhook URL (looks like
   `https://hooks.slack.com/services/T000/B000/XXXXXXXX`).
5. Set it as `SLACK_WEBHOOK_URL` in your `.env` and as a GitHub secret (for
   CI) and pass it to `docker compose` / the EC2 container.

### 7.2 The 7 required alerts (all defined in `prometheus/alert_rules.yml`)

| # | Alert | Trigger | Slack Message |
|---|-------|---------|----------------|
| 1 | `DataLakeUnavailable` | `increase(datalake_unavailable[1m]) > 0` | Data source returned 503. Check API availability. |
| 2 | `FeatureAdded` | `increase(feature_added[1m]) > 0` | New feature detected in schema. Retraining may be required. |
| 3 | `FeatureRemoved` | `increase(feature_removed[1m]) > 0` | Feature dropped from schema. Verify pipeline compatibility. |
| 4 | `DistributionDrift` | `distribution_drift_detected == 1` | Data distribution drift detected. Model may be stale. |
| 5 | `FeatureDriftDetected` | `distribution_drift_detected > 0` | Feature-level drift flagged. Investigate upstream data. |
| 6 | `HighResponseLatency` | `histogram_quantile(0.95, ...) > 1.0` | P95 response latency exceeded 1 second. |
| 7 | `LowModelAccuracy` | `model_accuracy < 0.8` | Model accuracy dropped below threshold. Auto-retraining triggered. |

### 7.3 Demonstrating each alert (for screenshots)

The inference API exposes a debug endpoint (for demo purposes only) that
flips the underlying metric values so each alert fires:

```bash
BASE=http://<EC2_PUBLIC_IP>:8000   # or http://localhost:8000

curl -X POST $BASE/debug/trigger/datalake          # Alert 1
curl -X POST $BASE/debug/trigger/feature_added     # Alert 2
curl -X POST $BASE/debug/trigger/feature_removed   # Alert 3
curl -X POST $BASE/debug/trigger/drift             # Alerts 4 & 5
curl -X POST $BASE/debug/trigger/latency           # Alert 6
curl -X POST $BASE/debug/trigger/low_accuracy      # Alert 7

# Reset state afterwards:
curl -X POST $BASE/debug/trigger/clear_drift
curl -X POST $BASE/debug/trigger/restore_accuracy
```

After each trigger, wait ~30–60s (Prometheus scrape interval +
Alertmanager group_wait) and check `#mlops-alerts` in Slack, then take a
screenshot and save it to `screenshots/`.

> ⚠️ Remove or protect the `/debug/trigger/*` endpoint before any real
> production deployment — it is included here solely to satisfy the
> "demonstrate all 7 alerts" deliverable.

---

## 8. CI/CD — GitHub Actions (Part 7)

### 8.1 Configure repository secrets

GitHub repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value |
|---|---|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub access token |
| `EC2_SSH_KEY` | Full contents of your EC2 `.pem` private key |
| `EC2_HOST` | EC2 public IP or DNS |
| `EC2_USER` | `ubuntu` |
| `SLACK_WEBHOOK_URL` | Your Slack incoming webhook URL |

### 8.2 Pipeline jobs (`.github/workflows/mlops-ci.yml`)

1. **lint-and-test** — flake8, trains a model (so `/predict` tests work),
   runs pytest, and (bonus) fails the build if accuracy < 0.80.
2. **build-and-push** — builds the Docker image and pushes
   `:latest` and `:<git-sha>` tags to Docker Hub.
3. **deploy** — SSHes into EC2, pulls the new image, restarts the
   container, and verifies `/health` returns 200.

Push to `main` to trigger the pipeline:

```bash
git add .
git commit -m "Initial MLOps pipeline"
git push origin main
```

Check the **Actions** tab in GitHub for a green run across all 3 jobs.

---

## 9. End-to-End Run Order (Quick Reference)

```bash
# 1. Local setup
pip install -r requirements.txt
cp .env.example .env   # fill in SLACK_WEBHOOK_URL

# 2. Ingest + train
python ingestion/ingestion.py --once
python model/train.py

# 3. Serve locally and smoke-test
uvicorn serving.app:app --host 0.0.0.0 --port 8000 &
curl localhost:8000/health
curl localhost:8000/metrics

# 4. Tests
pytest tests/ -v
flake8 ingestion model serving exporter tests

# 5. Build/push/deploy to EC2
export DOCKER_USERNAME=... DOCKER_PASSWORD=... DOCKER_IMAGE_NAME=...
export EC2_HOST=... EC2_USER=ubuntu EC2_SSH_KEY_PATH=./key.pem SLACK_WEBHOOK_URL=...
./deploy/deploy.sh all

# 6. Observability stack
# (edit prometheus/prometheus.yml with your EC2 IP first)
docker compose up -d
# visit localhost:9090 (Prometheus), localhost:3000 (Grafana), localhost:9093 (Alertmanager)

# 7. Trigger alerts for screenshots
curl -X POST http://<EC2_HOST>:8000/debug/trigger/low_accuracy
# ... check Slack #mlops-alerts

# 8. CI/CD
git push origin main   # GitHub Actions runs lint/test -> build/push -> deploy
```

---

## 10. Video Demo

`<Insert your Google Drive / YouTube unlisted link here>`

The video covers: data ingestion run, a firing Slack alert, the Grafana
dashboard with live data, and a passing GitHub Actions CI/CD run.

---

## 11. Notes on Security

- No credentials are hardcoded anywhere in this repo.
- `.env` is gitignored; only `.env.example` (with placeholders) is committed.
- All secrets (`DOCKER_USERNAME`, `DOCKER_PASSWORD`, `EC2_SSH_KEY`,
  `EC2_HOST`, `EC2_USER`, `SLACK_WEBHOOK_URL`) are stored as GitHub repository
  secrets and injected at CI/CD runtime only.
- The Alertmanager Slack webhook is injected via environment variable
  substitution at container start (`alertmanager/entrypoint.sh`), never
  hardcoded in `alertmanager.yml.template`.
