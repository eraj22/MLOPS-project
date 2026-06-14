FROM python:3.11-slim

WORKDIR /app

# System deps (for scikit-learn / numpy wheels build speed-up, optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY exporter/ ./exporter/
COPY ingestion/ ./ingestion/
COPY model/ ./model/
COPY serving/ ./serving/

# Ensure a model exists at build time (fallback synthetic training if none provided)
RUN python model/train.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
