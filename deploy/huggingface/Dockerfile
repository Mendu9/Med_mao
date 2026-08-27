# MAO — HuggingFace Spaces, Docker SDK variant.
#
# Build from the REPO ROOT so the whole source tree is in context:
#   docker build -f deploy/huggingface/Dockerfile -t mao-hf .
#
# This target is only used when the Space is configured with `sdk: docker`.
# The Space is currently `sdk: streamlit` (README front-matter), which runs
# app.py directly and never reads this file. See deploy/README.md.
#
# Fixed here (P1-15): the previous Dockerfile.hf COPYd scripts/start_hf.sh,
# which .gitignore excluded, so it could not build from a clean clone. The
# launcher now lives next to this file and is tracked.
FROM python:3.11-slim

# System deps: build-essential/libpq-dev for source builds, curl for the
# readiness probe in start_hf.sh, ffmpeg only if the audio extra is installed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first for layer caching. requirements.txt is authoritative for
# every target; extras stay opt-in (see requirements-optional.txt).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MAO_API_URL=http://localhost:8080 \
    MAO_API_PORT=8080 \
    PORT=7860 \
    HF_HOME=/tmp/hf_cache

RUN chmod +x deploy/huggingface/start_hf.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["./deploy/huggingface/start_hf.sh"]
