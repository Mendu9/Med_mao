#!/bin/bash
# MAO startup script — runs FastAPI backend + Streamlit frontend in one container.
# HF Spaces exposes port 7860 publicly; FastAPI runs internally on 8080.

set -e

# Create model cache dir if using persistent storage
mkdir -p "${HF_HOME:-/data/hf_cache}"

# Start FastAPI backend in background
uvicorn mao.api.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1 \
    --log-level info &

FASTAPI_PID=$!
echo "FastAPI started (PID $FASTAPI_PID) on port 8080"

# Wait for FastAPI to be ready (up to 60s)
for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        echo "FastAPI ready after ${i}s"
        break
    fi
    sleep 2
done

# Start Streamlit on port 7860 (HF Spaces public port) — foreground process
exec streamlit run app/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
