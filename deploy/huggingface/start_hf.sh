#!/bin/bash
# Two-process launcher for a Docker-SDK HuggingFace Space:
#   FastAPI (uvicorn) on 8080, Streamlit on 7860.
#
# Used only by deploy/huggingface/Dockerfile. The default Streamlit-SDK Space
# does NOT use this script — it runs app.py, which starts the API in a thread.
# See deploy/README.md for which entrypoint is authoritative for which target.
#
# This file replaces the untracked scripts/start_hf.sh and the root start.sh,
# which had drifted into two divergent copies of the same launcher (P1-15/P1-16).
set -euo pipefail

echo "===== MAO startup at $(date -u '+%Y-%m-%d %H:%M:%S') ====="

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export DATA_DIR="${DATA_DIR:-/app/mao/data}"
API_PORT="${MAO_API_PORT:-8080}"
UI_PORT="${PORT:-7860}"

mkdir -p "$HF_HOME" "$DATA_DIR"

echo "HF_HOME  : $HF_HOME"
echo "DATA_DIR : $DATA_DIR"
echo "API_PORT : $API_PORT"
echo "UI_PORT  : $UI_PORT"

uvicorn mao.api.main:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --workers 1 \
    --log-level info &
API_PID=$!
echo "FastAPI started (PID $API_PID) on port $API_PORT"

# Stop the API whenever this script exits, however it exits.
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 30); do
    if curl -sf "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        echo "FastAPI ready after $((i * 2))s"
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "FastAPI exited during startup" >&2
        exit 1
    fi
    sleep 2
done

exec streamlit run app/streamlit_app.py \
    --server.port "$UI_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.enableCORS false \
    --server.enableXsrfProtection false
