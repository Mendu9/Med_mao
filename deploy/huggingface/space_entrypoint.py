"""HuggingFace Spaces entrypoint — deployment glue only.

This is the target-specific half of what used to live in the root ``app.py``:
free-tier feature toggles, cache locations, the in-process FastAPI thread and
the Qdrant keep-alive. It contains no application logic; everything it starts
comes from ``mao.*`` and ``app.streamlit_app`` unchanged, so no core behaviour
can diverge per deployment target (P1-16).

The root ``app.py`` is a shim that calls :func:`main` — HF Spaces reads
``app_file: app.py`` from the README front-matter and that must keep resolving.
"""
from __future__ import annotations

import logging
import os
import runpy
import socket
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

API_PORT = int(os.getenv("MAO_API_PORT", "8080"))
_API_BIND_GRACE_SECONDS = 5
_QDRANT_KEEPALIVE_SECONDS = 23 * 3600


def configure_environment() -> None:
    """Apply HF free-tier defaults. Every one is overridable via Space secrets."""
    # Heavy components that are unavailable or unaffordable on the free tier.
    os.environ.setdefault("MAO_DISABLE_MEM0", "1")      # Mem0 needs a local ChromaDB service
    os.environ.setdefault("MAO_DISABLE_GRAPH", "0")     # graph traversal enabled
    os.environ.setdefault("MAO_DISABLE_BM25", "1")      # 154K-doc scoring is 4+ min/query on free CPU
    os.environ.setdefault("MAO_DISABLE_RERANKER", "0")  # reranker enabled

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")

    # /tmp is the only reliably writable path on a Space.
    os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/hf_cache/sentence_transformers")
    os.environ.setdefault("TORCH_HOME", "/tmp/hf_cache/torch")


def port_in_use(port: int) -> bool:
    """Return True if something is already listening on the given TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _serve_api() -> None:
    import uvicorn

    from mao.api.main import app as fastapi_app

    config = uvicorn.Config(
        fastapi_app,
        host="0.0.0.0",
        port=API_PORT,
        log_level="warning",
        loop="asyncio",  # explicit loop — avoids uvloop import issues on HF
    )
    uvicorn.Server(config).run()


def _qdrant_keepalive() -> None:
    """Ping Qdrant every 23 hours so the free cluster never goes idle."""
    while True:
        time.sleep(_QDRANT_KEEPALIVE_SECONDS)
        try:
            from qdrant_client import QdrantClient

            from mao.core.config import cfg

            if cfg.qdrant_url and cfg.qdrant_api_key:
                client = QdrantClient(
                    url=cfg.qdrant_url,
                    api_key=cfg.qdrant_api_key,
                    prefer_grpc=False,
                    timeout=10,
                )
                info = client.get_collections()
                logger.info("Qdrant keep-alive ping OK (%d collections)", len(info.collections))
        except Exception as exc:  # noqa: BLE001 - a keep-alive must never kill the Space
            logger.warning("Qdrant keep-alive ping failed (cluster may be suspended): %s", exc)


def start_background_services() -> bool:
    """Start the in-process FastAPI thread and the Qdrant keep-alive.

    Returns False when the API port is already bound, which means Streamlit
    hot-reloaded this module rather than booting a fresh container.
    """
    if port_in_use(API_PORT):
        logger.info("FastAPI already running on port %d — skipping thread start.", API_PORT)
        return False

    threading.Thread(target=_serve_api, daemon=True, name="fastapi").start()
    # Give the API a moment to bind before Streamlit starts making requests.
    time.sleep(_API_BIND_GRACE_SECONDS)
    threading.Thread(target=_qdrant_keepalive, daemon=True, name="qdrant-keepalive").start()
    return True


def main() -> None:
    """Boot the Space: FastAPI in a background thread, then Streamlit in-process."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    configure_environment()
    start_background_services()
    runpy.run_module("app.streamlit_app", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
