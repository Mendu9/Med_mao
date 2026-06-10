"""MAO Clinical AI — HuggingFace Spaces entry point."""
import runpy, sys, os, threading, time, socket

sys.path.insert(0, os.path.dirname(__file__))

# Disable heavy components that aren't available on HF free tier by default.
# Users can override these via HF Space secrets.
os.environ.setdefault("MAO_DISABLE_MEM0", "1")    # Mem0 needs ChromaDB (local service) — disabled by default; set 0 + CHROMA_* secrets to enable
os.environ.setdefault("MAO_DISABLE_GRAPH", "0")   # Graph enabled
os.environ.setdefault("MAO_DISABLE_BM25", "0")    # corpus downloaded from HF Hub dataset ArunMendu/Med_mao-data on first use
os.environ.setdefault("MAO_DISABLE_RERANKER", "0")  # Reranker enabled

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/hf_cache/sentence_transformers")
os.environ.setdefault("TORCH_HOME", "/tmp/hf_cache/torch")


def _port_in_use(port: int) -> bool:
    """Return True if something is already listening on the given TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_api():
    import uvicorn
    from mao.api.main import app as fastapi_app
    config = uvicorn.Config(
        fastapi_app,
        host="0.0.0.0",
        port=8080,
        log_level="warning",
        loop="asyncio",      # explicit loop — avoids uvloop import issues on HF
    )
    server = uvicorn.Server(config)
    server.run()


def _qdrant_keepalive():
    """Ping Qdrant every 23 hours so the free cluster never goes idle."""
    import logging
    _log = logging.getLogger(__name__)
    while True:
        time.sleep(23 * 3600)
        try:
            from mao.core.config import cfg
            from qdrant_client import QdrantClient
            if cfg.qdrant_url and cfg.qdrant_api_key:
                c = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, prefer_grpc=False, timeout=10)
                info = c.get_collections()
                _log.info("Qdrant keep-alive ping OK (%d collections)", len(info.collections))
        except Exception as exc:
            _log.warning("Qdrant keep-alive ping failed (cluster may be suspended): %s", exc)


# On HF Spaces the FastAPI backend runs in a background thread on port 8080.
# Streamlit talks to it via http://localhost:8080 (the default MAO_API_URL).
# Guard against Streamlit hot-reload re-executing this module: only start the
# API thread when the port is not already bound (first boot).
if not _port_in_use(8080):
    api_thread = threading.Thread(target=_start_api, daemon=True, name="fastapi")
    api_thread.start()
    # Give the API a moment to bind before Streamlit starts making requests
    time.sleep(5)

    # Keep the Qdrant free cluster alive with a daily ping (starts after 23 h)
    keepalive_thread = threading.Thread(target=_qdrant_keepalive, daemon=True, name="qdrant-keepalive")
    keepalive_thread.start()
else:
    # Port already bound — Streamlit hot-reload, skip re-starting the API.
    import logging
    logging.getLogger(__name__).info("FastAPI already running on port 8080 — skipping thread start.")

runpy.run_module("app.streamlit_app", run_name="__main__", alter_sys=True)
