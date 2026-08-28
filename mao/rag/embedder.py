"""
mao/rag/embedder.py — sentence-transformers embeddings (local, no API key).
Model: configured via EMBED_MODEL env var (default: NeuML/pubmedbert-base-embeddings, 768-dim).
"""
from __future__ import annotations
import logging
from mao.core.config import cfg

logger = logging.getLogger(__name__)
_model = None


def get_embedding_model():
    """The process-wide sentence-transformer instance.

    Public because it is the single owner: `mao.rag.chunker` needs the same
    model for boundary similarity, and building its own meant loading the same
    weights into memory a second time.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", cfg.embed_model)
        _model = SentenceTransformer(cfg.embed_model)
    return _model


def _get_model():
    """Internal alias retained for existing call sites in this module."""
    return get_embedding_model()


def embed_query(text: str) -> list[float]:
    # .flatten() ensures 1-D output even when the model returns a 2-D array for a single string
    return _get_model().encode(text, convert_to_numpy=True).flatten().tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, convert_to_numpy=True).tolist()
