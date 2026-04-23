"""
mao/rag/embedder.py — sentence-transformers embeddings (local, no API key).
Model: all-MiniLM-L6-v2  ->  384-dim, fast, works on HuggingFace Spaces.
"""
from __future__ import annotations
import logging
from mao.core.config import cfg

logger = logging.getLogger(__name__)
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", cfg.embed_model)
        _model = SentenceTransformer(cfg.embed_model)
    return _model


def embed_query(text: str) -> list[float]:
    return _get_model().encode(text, convert_to_numpy=True).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, convert_to_numpy=True).tolist()
