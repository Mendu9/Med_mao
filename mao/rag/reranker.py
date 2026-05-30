"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3 via FlagEmbedding; called at step 9 of the retrieval pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

# FlagEmbedding import is deferred to _get_reranker() — importing it at module
# level triggers TF+PyTorch init which causes a 3-minute hang on Windows.
FlagReranker = None  # type: ignore[assignment,misc]
_FLAG_AVAILABLE: bool | None = None  # None = not yet checked

from mao.core.config import cfg

logger = logging.getLogger(__name__)

_MAX_BATCH = 64  # prevent OOM on large inputs


@dataclass
class RankedChunk:
    """A document chunk with its reranker relevance score."""
    text: str
    score: float
    metadata: dict


# Lazy-load: the model is ~2GB, loaded on first call to _get_reranker()
_reranker: "object | None" = None


_reranker_failed: bool = False


def _get_reranker():  # type: ignore[return]
    global _reranker, _reranker_failed, FlagReranker, _FLAG_AVAILABLE
    import os
    if os.getenv("MAO_DISABLE_RERANKER", "0") == "1":
        return None
    if _reranker_failed:
        return None
    # Lazy import — avoids TF+PyTorch init at module load time
    if _FLAG_AVAILABLE is None:
        try:
            from FlagEmbedding import FlagReranker as _FR
            FlagReranker = _FR
            _FLAG_AVAILABLE = True
        except ImportError:
            _FLAG_AVAILABLE = False
    if not _FLAG_AVAILABLE:
        return None
    if _reranker is None:
        try:
            logger.info("Loading reranker model: %s", cfg.reranker_model)
            _reranker = FlagReranker(cfg.reranker_model, use_fp16=True)
            logger.info("Reranker loaded.")
        except Exception as exc:
            logger.warning("Reranker unavailable (%s) — using score passthrough", exc)
            _reranker_failed = True
            return None
    return _reranker


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[RankedChunk]:
    """
    Rerank *chunks* against *query* using the cross-encoder.

    Args:
        query:  The user query string.
        chunks: List of dicts, each with at least a "text" key.
                Additional keys (metadata, source, etc.) are preserved.
        top_k:  How many top results to return. Defaults to cfg.reranker_top_k.

    Returns:
        List of RankedChunk sorted by score descending, length = top_k.

    This is a synchronous call (FlagReranker is not async-native).
    Run in a thread executor if called from async context:
        loop.run_in_executor(None, rerank, query, chunks)
    """
    if not chunks:
        return []

    k = top_k if top_k is not None else cfg.reranker_top_k
    reranker = _get_reranker()

    if reranker is None:
        # Sort by existing cosine score so best results come first
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
        return [
            RankedChunk(text=c.get("text", ""), score=c.get("score", 0.0), metadata=c)
            for c in sorted_chunks[:k]
        ]

    texts = [c.get("text", "") for c in chunks]

    # FlagReranker expects list of [query, passage] pairs
    pairs = [[query, t] for t in texts]

    try:
        _t0 = time.monotonic()
        all_scores: list[float] = []
        for i in range(0, len(pairs), _MAX_BATCH):
            batch = pairs[i : i + _MAX_BATCH]
            raw = reranker.compute_score(batch, normalize=True)
            # Some FlagReranker versions return a scalar float for single-pair batches
            if isinstance(raw, (int, float)):
                raw = [float(raw)]
            all_scores.extend(raw)
        scores: list[float] = all_scores
        elapsed_ms = (time.monotonic() - _t0) * 1000
        logger.debug("Reranker scored %d pairs in %.0fms", len(pairs), elapsed_ms)
        # Release CPU tensor allocations after each inference pass
        import gc as _gc
        _gc.collect()
    except Exception as exc:  # noqa: BLE001
        logger.error("Reranker inference failed: %s", exc)
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
        return [
            RankedChunk(text=c.get("text", ""), score=c.get("score", 0.0), metadata=c)
            for c in sorted_chunks[:k]
        ]

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        RankedChunk(
            text=chunk.get("text", ""),
            score=float(score),
            metadata=chunk,
        )
        for score, chunk in ranked[:k]
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_chunks = [
        {"text": "GraphRAG combines vector search with knowledge graphs.", "source": "doc1"},
        {"text": "Python is a general-purpose programming language.", "source": "doc2"},
        {"text": "Knowledge graphs store entities and their relationships.", "source": "doc3"},
    ]
    results = rerank("What is GraphRAG?", sample_chunks, top_k=2)
    for r in results:
        print(f"Score={r.score:.4f} | {r.text[:80]}")
