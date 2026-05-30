"""MAO Clinical AI — HuggingFace Spaces entry point."""
import runpy, sys, os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

# Disable heavy components that aren't available on HF free tier by default.
# Users can override these via HF Space secrets.
os.environ.setdefault("MAO_DISABLE_MEM0", "1")    # Mem0 needs ChromaDB (local service) — disabled by default; set 0 + CHROMA_* secrets to enable
os.environ.setdefault("MAO_DISABLE_GRAPH", "0")   # Graph enabled
os.environ.setdefault("MAO_DISABLE_BM25", "1")    # BM25 corpus (95 MB) excluded from HF Spaces; vector-only retrieval used instead
os.environ.setdefault("MAO_DISABLE_RERANKER", "0")  # Reranker enabled

# Prevent TensorFlow import hang (FlagEmbedding / reranker)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# Point HF model cache to a writable location on HF Spaces
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/hf_cache/sentence_transformers")
os.environ.setdefault("TORCH_HOME", "/tmp/hf_cache/torch")

runpy.run_module("app.streamlit_app", run_name="__main__", alter_sys=True)
