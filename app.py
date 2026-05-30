"""MAO Clinical AI — HuggingFace Spaces entry point."""
import runpy, sys, os, threading, time

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

# On HF Spaces the FastAPI backend runs in a background thread on port 8080.
# Streamlit talks to it via http://localhost:8080 (the default MAO_API_URL).
def _start_api():
    import uvicorn
    from mao.api.main import app as fastapi_app
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8080, log_level="warning")

api_thread = threading.Thread(target=_start_api, daemon=True, name="fastapi")
api_thread.start()

# Give the API a moment to bind before Streamlit starts making requests
time.sleep(3)

runpy.run_module("app.streamlit_app", run_name="__main__", alter_sys=True)
