#!/bin/bash
set -e

echo "===== Application Startup at $(date -u '+%Y-%m-%d %H:%M:%S') ====="

mkdir -p "${HF_HOME:-/home/user/.cache/huggingface}"
mkdir -p "${DATA_DIR:-/home/user/app/mao/data}"
export DATA_DIR="${DATA_DIR:-/home/user/app/mao/data}"

echo "HF_HOME  : ${HF_HOME:-/home/user/.cache/huggingface}"
echo "DATA_DIR : $DATA_DIR"

uvicorn mao.api.main:app --host 0.0.0.0 --port 8080 --workers 1 --log-level info &
FASTAPI_PID=$!
echo "FastAPI started (PID $FASTAPI_PID) on port 8080"

for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        echo "FastAPI ready after ${i}s"
        break
    fi
    sleep 2
done

exec streamlit run app/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
