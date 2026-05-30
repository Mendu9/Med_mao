# MAO API — multi-stage build
FROM python:3.11-slim AS base

# System deps for spaCy, psycopg2, whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "gunicorn>=21.2"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source
COPY . .

# Install project as package
RUN pip install --no-cache-dir -e .

EXPOSE 8080

# Multi-worker: gunicorn manages N worker processes, each running a uvicorn event loop.
# Default workers = (2 * CPU_COUNT + 1); override with MAO_WORKERS env var.
# Use MAO_DISABLE_RERANKER=1 to save ~568 MB RAM per worker on constrained hosts.
CMD ["gunicorn", "mao.api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--config", "gunicorn.conf.py", \
     "--bind", "0.0.0.0:8080"]
