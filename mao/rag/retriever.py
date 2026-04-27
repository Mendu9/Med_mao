"""
mao/rag/retriever.py
--------------------
GraphRAG retrieval pipeline — the core intellectual asset of MAO.

5-step pipeline (MUST follow exactly):
  1. Vector search      → top-N candidates from ChromaDB
  2. Entity extraction  → NER on top-N results
  3. Graph traversal    → expand via NetworkX entity links
  4. Fetch graph chunks → retrieve ChromaDB chunks for expanded entities
  5. Merge + dedup      → combine vector + graph chunks
  6. Reranker           → bge-reranker-v2-m3, return top-K

Why hybrid vector+graph over pure vector:
  - Vector search misses multi-hop reasoning ("who advised X's advisor?")
  - Graph traversal surfaces entity neighbours that share no lexical overlap
  - Reranker then filters noise, giving precision on top of graph recall

Integration points:
  - agents/graphrag_agent.py  calls retrieve() as its primary operation
  - data/ingest_wikipedia.py  populates the ChromaDB collection this queries
  - rag/graph_builder.py      provides expand_via_graph()
  - rag/reranker.py           provides rerank()
  - rag/embedder.py           provides embed_query()
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from mao.core.config import cfg, CACHE_TTL
from mao.core.query_cache import QueryCache
from mao.rag.embedder import embed_query
from mao.rag.graph_builder import extract_entities, expand_via_graph, load_graph
from mao.rag.reranker import RankedChunk, rerank

_cache = QueryCache(ttl=CACHE_TTL)
_chroma_client: chromadb.HttpClient | None = None
_chroma_collection = None


def _get_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
        _chroma_collection = _chroma_client.get_or_create_collection(cfg.chroma_collection)
        logger.info("ChromaDB collection '%s' ready (%d docs)", cfg.chroma_collection, _chroma_collection.count())
    return _chroma_collection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_n: int | None = None,
    top_k: int | None = None,
    domain: str = "alzheimer",
) -> list[RankedChunk]:
    """
    Execute the full 5-step GraphRAG retrieval pipeline.

    Args:
        query:   User query string.
        top_n:   Vector search candidate count (default: cfg.reranker_top_n = 20).
        top_k:   Final results after reranking (default: cfg.reranker_top_k = 5).
        domain:  Domain for index selection: "alzheimer" | "stroke" | "general".

    Returns:
        List of RankedChunk objects sorted by reranker score, length <= top_k.
    """
    n = top_n or cfg.reranker_top_n   # 20
    k = top_k or cfg.reranker_top_k   # 5

    # Check cache first
    try:
        query_embedding = embed_query(query)
        cache_key = _cache.make_key(query_embedding + [hash(domain) % 1_000_000, n, k])
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for query domain=%s", domain)
            return cached
    except Exception:
        query_embedding = None
        cache_key = None

    # ------------------------------------------------------------------
    # Step 1: Vector search → top-N from ChromaDB
    # ------------------------------------------------------------------
    vector_chunks = _vector_search(query, n)
    logger.debug("Step 1 vector search: %d results", len(vector_chunks))

    # ------------------------------------------------------------------
    # Step 2: Entity extraction on retrieved texts
    # ------------------------------------------------------------------
    all_text = " ".join(c.get("text", "") for c in vector_chunks)
    seed_entities = [ent for ent, _ in extract_entities(all_text)]
    logger.debug("Step 2 extracted entities: %s", seed_entities[:10])

    # ------------------------------------------------------------------
    # Step 3: Graph traversal → find related entity names
    # ------------------------------------------------------------------
    graph = load_graph()
    expanded_entities: list[str] = []
    if graph.number_of_nodes() > 0 and seed_entities:
        expanded_entities = expand_via_graph(graph, seed_entities, hops=2, max_nodes=30)
    logger.debug("Step 3 expanded %d entity neighbours", len(expanded_entities))

    # ------------------------------------------------------------------
    # Step 4: Fetch ChromaDB chunks mentioning expanded entities
    # ------------------------------------------------------------------
    graph_chunks: list[dict[str, Any]] = []
    if expanded_entities:
        graph_chunks = _entity_search(expanded_entities, limit=n)
    logger.debug("Step 4 graph-derived chunks: %d", len(graph_chunks))

    # ------------------------------------------------------------------
    # Step 5: Merge + deduplicate
    # ------------------------------------------------------------------
    merged = _merge_deduplicate(vector_chunks, graph_chunks)
    logger.debug("Step 5 merged+deduped: %d chunks", len(merged))

    # ------------------------------------------------------------------
    # Step 6: Reranker → top-K
    # ------------------------------------------------------------------
    ranked = rerank(query, merged, top_k=k)
    logger.info(
        "GraphRAG pipeline complete: %d → %d final chunks (top_k=%d)",
        len(merged),
        len(ranked),
        k,
    )

    # Store in cache
    if cache_key is not None:
        try:
            _cache.set(cache_key, ranked)
        except Exception:
            pass

    return ranked


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vector_search(query: str, n: int) -> list[dict[str, Any]]:
    """Embed query and fetch top-n chunks from ChromaDB."""
    try:
        col = _get_collection()
        query_embedding = embed_query(query)
        results = col.query(query_embeddings=[query_embedding], n_results=min(n, col.count()))
        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc, meta, cid, dist in zip(docs, metas, ids, distances):
            meta = meta or {}
            chunks.append({
                "text": doc,
                "source": meta.get("source", ""),
                "chunk_id": meta.get("chunk_id", cid),
                "score": 1.0 - dist,
                "retrieval_method": "vector",
                **meta,
            })
        return chunks
    except Exception as exc:
        logger.error("ChromaDB vector search failed: %s", exc)
        return []


def _entity_search(entities: list[str], limit: int) -> list[dict[str, Any]]:
    """Query ChromaDB with entity names to find related chunks."""
    try:
        col = _get_collection()
        entity_query = " ".join(entities[:10])
        query_embedding = embed_query(entity_query)
        results = col.query(query_embeddings=[query_embedding], n_results=min(limit, col.count()))
        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc, meta, cid, dist in zip(docs, metas, ids, distances):
            meta = meta or {}
            chunks.append({
                "text": doc,
                "source": meta.get("source", ""),
                "chunk_id": meta.get("chunk_id", cid),
                "score": 1.0 - dist,
                "retrieval_method": "graph",
                **meta,
            })
        return chunks
    except Exception as exc:
        logger.error("Entity search failed: %s", exc)
        return []


def _merge_deduplicate(
    vector_chunks: list[dict[str, Any]],
    graph_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge two chunk lists, deduplicating by chunk_id (fallback: text prefix).
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    for chunk in vector_chunks + graph_chunks:
        key = chunk.get("chunk_id") or chunk.get("text", "")[:100]
        if key and key not in seen:
            seen.add(key)
            merged.append(chunk)

    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    results = retrieve("What is the capital of France?")
    for r in results:
        print(f"[{r.score:.4f}] {r.text[:120]}")
