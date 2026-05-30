"""
gunicorn.conf.py — MAO multi-worker deployment configuration.

Workers default to (2 * CPU_COUNT + 1). Override with MAO_WORKERS env var.
Each worker loads the LangGraph graph + embedding model independently.
Full stack (reranker + embedder + graph): ~2-3 GB/worker.
Set MAO_DISABLE_RERANKER=1 to reduce to ~1.5 GB/worker.
"""
import multiprocessing
import os

workers      = int(os.getenv("MAO_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
keepalive    = 75
max_requests        = 1000
max_requests_jitter = 100   # stagger restarts to avoid thundering herd
graceful_timeout    = 120
timeout             = 180
bind         = f"0.0.0.0:{os.getenv('API_PORT', '8080')}"
preload_app  = os.getenv("MAO_PRELOAD_APP", "1") == "1"
access_log_format = (
    '{"ts":"%(t)s","method":"%(m)s","path":"%(U)s",'
    '"status":%(s)s,"bytes":%(b)s,"duration_us":%(D)s}'
)
