# MAO API — self-hosted container. API ONLY (no Streamlit UI).
#
#   docker build -t mao-api .
#   docker run -p 8080:8080 --env-file .env mao-api
#
# For the HuggingFace Spaces targets see deploy/huggingface/ and deploy/README.md.
FROM python:3.11-slim AS base

# System deps: build-essential/libpq-dev for any source builds, curl for the
# healthcheck. ffmpeg is only needed by the optional audio extra
# (openai-whisper, see requirements-optional.txt) and is installed with it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "gunicorn>=21.2"

# NOTE: the spaCy model is NOT downloaded with `python -m spacy download` here.
# That line broke this build when spacy was dropped from requirements.txt
# (P1-15). The model is now an ordinary pinned wheel in requirements.txt, so it
# installs with everything else and needs no post-install network step.

# Copy source
COPY . .

# Install project as package
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

# Multi-worker: gunicorn manages N worker processes, each running a uvicorn event loop.
# Default workers = (2 * CPU_COUNT + 1); override with MAO_WORKERS env var.
# Use MAO_DISABLE_RERANKER=1 to save ~568 MB RAM per worker on constrained hosts.
CMD ["gunicorn", "mao.api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--config", "gunicorn.conf.py", \
     "--bind", "0.0.0.0:8080"]
