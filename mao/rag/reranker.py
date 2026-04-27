"""
mao/rag/reranker.py
-------------------
Cross-encoder reranker using BAAI/bge-reranker-v2-m3 via FlagEmbedding.

Why bge-reranker-v2-m3:
  - State-of-the-art on MTEB reranking benchmark (as of mid-2024)
  - Runs on CPU with use_fp16=True — no GPU required
  - Used in production RAG pipelines at scale
  - FlagEmbedding library is actively maintained by BAAI

Pipeline position:
  Step 4 of 5 in GraphRAG retrieval:
  [vector search (20)] → [graph expand] → [merge/dedup] → [RERANK (top 5)] → [LLM]

Integration points:
  - rag/retriever.py calls rerank() after merge+dedup step
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from FlagEmbedding import FlagReranker  # pip install FlagEmbedding

from mao.core.config import cfg

logger = logging.getLogger(__name__)


@dataclass
class RankedChunk:
    """A document chunk with its reranker relevance score."""
    text: str
    score: float
    metadata: dict


# Lazy-load: the model is ~600 MB, load once at first use
_reranker: FlagReranker | None = None


_reranker_failed: bool = False


def _get_reranker() -> FlagReranker | None:
    global _reranker, _reranker_failed
    if _reranker_failed:
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
        return [
            RankedChunk(text=c.get("text", ""), score=c.get("score", 0.0), metadata=c)
            for c in chunks[:k]
        ]

    texts = [c.get("text", "") for c in chunks]

    # FlagReranker expects list of [query, passage] pairs
    pairs = [[query, t] for t in texts]

    try:
        scores: list[float] = reranker.compute_score(pairs, normalize=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Reranker inference failed: %s", exc)
        # Fallback: return top-k chunks in original order
        return [
            RankedChunk(text=c.get("text", ""), score=0.0, metadata=c)
            for c in chunks[:k]
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
