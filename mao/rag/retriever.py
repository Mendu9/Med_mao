"""9-step GraphRAG retrieval pipeline: query expansion, BM25, vector search, RRF, graph traversal, reranker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import chromadb

from mao.core.config import cfg, CACHE_TTL
from mao.core.query_cache import QueryCache
from mao.rag.embedder import embed_query
from mao.rag.graph_builder import extract_entities, expand_via_graph, load_graph
from mao.rag.reranker import RankedChunk, rerank

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

_cache = QueryCache(ttl=CACHE_TTL)
_chroma_client: chromadb.HttpClient | None = None
_chroma_collection = None
_qdrant_client: Any | None = None  # lazy singleton for Qdrant backend

# ---------------------------------------------------------------------------
# BM25 module-level state (lazy initialised via _build_bm25_index)
# ---------------------------------------------------------------------------
_bm25_index: "BM25Okapi | None" = None
_bm25_corpus: list[str] = []
_bm25_chunk_ids: list[str] = []

# Path where the BM25 index is persisted between restarts
_BM25_PICKLE_PATH = cfg.data_dir / "bm25_index.pkl"

logger = logging.getLogger(__name__)


_bm25_loaded: bool = False


def _load_bm25_from_disk() -> None:
    """Load BM25 index from disk, called lazily on first _bm25_search() call.

    Deferred from module import so BM25 (~1.5GB) and the reranker (~2.2GB)
    don't both load simultaneously during startup on memory-constrained hosts.
    Set MAO_DISABLE_BM25=1 to skip entirely (saves ~1.5GB; degrades MRR).
    """
    global _bm25_index, _bm25_corpus, _bm25_chunk_ids, _bm25_loaded
    if _bm25_loaded:
        return
    _bm25_loaded = True  # mark before load so concurrent callers don't double-load
    import os
    if os.getenv("MAO_DISABLE_BM25", "0") == "1":
        logger.info("BM25 index loading skipped (MAO_DISABLE_BM25=1)")
        return
    try:
        import pickle
        from rank_bm25 import BM25Okapi  # noqa: F401 — ensure importable
        if _BM25_PICKLE_PATH.exists():
            with open(_BM25_PICKLE_PATH, "rb") as fh:
                saved = pickle.load(fh)
            _bm25_index = saved["index"]
            _bm25_corpus = saved["corpus"]
            _bm25_chunk_ids = saved["chunk_ids"]
            logger.info(
                "BM25 index loaded from disk: %d documents (%s)",
                len(_bm25_corpus), _BM25_PICKLE_PATH,
            )
    except Exception as exc:
        logger.debug("BM25 disk load skipped (%s)", exc)


def _get_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
        _chroma_collection = _chroma_client.get_or_create_collection(cfg.chroma_collection)
        logger.info("ChromaDB collection '%s' ready (%d docs)", cfg.chroma_collection, _chroma_collection.count())
    return _chroma_collection


def _get_qdrant_client():
    """Return lazy Qdrant client singleton. Raises if credentials missing."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client") from exc
        if not cfg.qdrant_url or not cfg.qdrant_api_key:
            raise ValueError(
                "VECTOR_BACKEND=qdrant requires QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY in .env"
            )
        _qdrant_client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
        count = _qdrant_client.count(cfg.qdrant_collection).count
        logger.info("Qdrant collection '%s' ready (%d points)", cfg.qdrant_collection, count)
    return _qdrant_client


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
    Execute the full 9-step GraphRAG retrieval pipeline (Sprint 3).

    Steps: query expansion → dense vector → BM25 sparse → RRF merge →
           entity extraction → graph traversal → entity search →
           final merge+dedup → reranker.

    Args:
        query:   User query string.
        top_n:   Vector search candidate count (default: cfg.reranker_top_n, env RERANKER_TOP_N=100).
        top_k:   Final results after reranking (default: cfg.reranker_top_k, env RERANKER_TOP_K=10).
        domain:  Domain for index selection: "alzheimer" | "stroke" | "general".

    Returns:
        List of RankedChunk objects sorted by reranker score, length <= top_k.
    """
    import os as _os
    _reranker_off = _os.getenv("MAO_DISABLE_RERANKER", "0") == "1"
    # When the reranker is disabled, fetch more candidates so raw cosine
    # similarity has a bigger pool to pick the best chunks from.
    default_n = 40 if _reranker_off else cfg.reranker_top_n
    n = top_n or default_n
    k = top_k or cfg.reranker_top_k

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
    # Step 1: Query expansion — ontology synonyms
    # ------------------------------------------------------------------
    expanded_query = _expand_query(query)
    logger.debug("Step 1 expanded query: '%s'", expanded_query[:120])

    # ------------------------------------------------------------------
    # Step 2: Dense vector search → top-N from ChromaDB
    # ------------------------------------------------------------------
    vector_chunks = _vector_search(expanded_query, n)
    logger.debug("Step 2 vector search: %d results", len(vector_chunks))

    # ------------------------------------------------------------------
    # Step 3: BM25 sparse search (degrades to [] if index not built)
    # ------------------------------------------------------------------
    bm25_results = _bm25_search(expanded_query, n=n)
    logger.debug("Step 3 BM25 search: %d results", len(bm25_results))

    # ------------------------------------------------------------------
    # Step 4: RRF merge — combine dense + sparse ranked lists
    # ------------------------------------------------------------------
    if bm25_results:
        merged_after_rrf = _reciprocal_rank_fusion([vector_chunks, bm25_results], k=60)
    else:
        merged_after_rrf = vector_chunks
    logger.debug("Step 4 RRF merge: %d chunks", len(merged_after_rrf))

    # ------------------------------------------------------------------
    # Step 5: Entity extraction on RRF-merged texts
    # NER load can crash (segfault on bad model state) — skip gracefully.
    # ------------------------------------------------------------------
    seed_entities: list[str] = []
    try:
        all_text = " ".join(c.get("text", "") for c in merged_after_rrf[:10])  # cap text length
        seed_entities = [ent for ent, _ in extract_entities(all_text)]
        logger.debug("Step 5 extracted entities: %s", seed_entities[:10])
    except Exception as exc:
        logger.debug("Step 5 entity extraction skipped (%s)", exc)

    # ------------------------------------------------------------------
    # Step 6: Graph traversal → find related entity names
    # ------------------------------------------------------------------
    graph = load_graph()
    expanded_entities: list[str] = []
    if graph.number_of_nodes() > 0 and seed_entities:
        expanded_entities = expand_via_graph(graph, seed_entities, hops=2, max_nodes=30)
    logger.debug("Step 6 expanded %d entity neighbours", len(expanded_entities))

    # ------------------------------------------------------------------
    # Step 7: Fetch ChromaDB chunks mentioning expanded entities
    #         (metadata filter preferred; vector fallback if no entities field)
    # ------------------------------------------------------------------
    graph_chunks: list[dict[str, Any]] = []
    if expanded_entities:
        graph_chunks = _entity_search(expanded_entities, limit=n)
    logger.debug("Step 7 graph-derived chunks: %d", len(graph_chunks))

    # ------------------------------------------------------------------
    # Step 8: Final merge + deduplicate all sources
    # ------------------------------------------------------------------
    merged = _merge_deduplicate(merged_after_rrf, graph_chunks)
    logger.debug("Step 8 merged+deduped: %d chunks", len(merged))

    # Boost score for chunks that appeared in both vector AND BM25 results
    # (cross-modal agreement signal — these chunks are more likely to be relevant)
    _vector_ids = {c.get("chunk_id") or c.get("text", "")[:50] for c in vector_chunks}
    _bm25_ids = {c.get("chunk_id") or c.get("text", "")[:50] for c in bm25_results}
    _both = _vector_ids & _bm25_ids
    for chunk in merged:
        cid = chunk.get("chunk_id") or chunk.get("text", "")[:50]
        if cid in _both:
            chunk["score"] = chunk.get("score", 0.5) * 1.15  # 15% boost
    logger.debug("Score-boosted %d chunks that appeared in both vector and BM25 results", len(_both))

    # ------------------------------------------------------------------
    # Step 9: Reranker → top-K
    # ------------------------------------------------------------------
    ranked = rerank(query, merged, top_k=k)

    # ------------------------------------------------------------------
    # Step 9b: MMR deduplication — remove near-duplicate chunks that
    # survived reranking (common when vector and BM25 return overlapping
    # paragraphs from the same source article).
    # ------------------------------------------------------------------
    ranked_deduped = _mmr_dedup(ranked, threshold=0.90)
    logger.info(
        "GraphRAG pipeline complete: %d → %d → %d final chunks (top_k=%d, after MMR dedup)",
        len(merged),
        len(ranked),
        len(ranked_deduped),
        k,
    )

    # Store in cache
    if cache_key is not None:
        try:
            _cache.set(cache_key, ranked_deduped)
        except Exception:
            pass

    return ranked_deduped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vector_search(query: str, n: int) -> list[dict[str, Any]]:
    """Embed query and fetch top-n chunks. Dispatches on cfg.vector_backend."""
    if cfg.vector_backend == "qdrant":
        return _vector_search_qdrant(query, n)
    return _vector_search_chroma(query, n)


def _vector_search_chroma(query: str, n: int) -> list[dict[str, Any]]:
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


def _vector_search_qdrant(query: str, n: int) -> list[dict[str, Any]]:
    try:
        client = _get_qdrant_client()
        query_embedding = embed_query(query)
        results = client.search(
            collection_name=cfg.qdrant_collection,
            query_vector=query_embedding,
            limit=n,
            with_payload=True,
        )
        chunks = []
        for hit in results:
            payload = hit.payload or {}
            chunks.append({
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "score": hit.score,
                "retrieval_method": "vector",
                **{k: v for k, v in payload.items() if k not in ("text",)},
            })
        return chunks
    except Exception as exc:
        logger.error("Qdrant vector search failed: %s", exc)
        return []


def _entity_search(entities: list[str], limit: int) -> list[dict[str, Any]]:
    """Fetch chunks mentioning entities. Dispatches on cfg.vector_backend."""
    if not entities:
        return []
    if cfg.vector_backend == "qdrant":
        return _entity_search_qdrant(entities, limit)
    return _entity_search_chroma(entities, limit)


def _entity_search_chroma(entities: list[str], limit: int) -> list[dict[str, Any]]:
    """ChromaDB entity search: vector search on entity name string.

    Uses entities as query text for vector search — the ChromaDB $contains
    operator is not supported in col.get(), so we use semantic search instead.
    """
    results: list[dict[str, Any]] = []
    try:
        col = _get_collection()
        entity_query = " ".join(entities[:10])
        query_embedding = embed_query(entity_query)
        res = col.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, col.count() or 1),
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for doc, meta, cid, dist in zip(docs, metas, ids, distances):
            meta = meta or {}
            results.append({
                "chunk_id": meta.get("chunk_id", cid),
                "text": doc,
                "score": 1.0 - dist,
                "source": meta.get("source", ""),
                "metadata": meta,
                "retrieval_method": "graph_vector",
            })
        logger.debug("Entity search (vector): %d chunks for %d entities", len(results), len(entities))
    except Exception as exc:
        logger.error("Entity search vector failed: %s", exc)

    return results


def _entity_search_qdrant(entities: list[str], limit: int) -> list[dict[str, Any]]:
    """Qdrant entity search: payload filter with vector fallback."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchText
        client = _get_qdrant_client()
        conditions = [
            FieldCondition(key="entities", match=MatchText(text=ent))
            for ent in entities[:5]
        ]
        hits = client.scroll(
            collection_name=cfg.qdrant_collection,
            scroll_filter=Filter(should=conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )[0]
        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "text": payload.get("text", ""),
                "score": 0.5,
                "source": payload.get("source", ""),
                "metadata": payload,
                "retrieval_method": "graph_metadata",
            })
        if results:
            logger.debug("Entity search Qdrant (payload filter): %d chunks", len(results))
            return results
    except Exception as exc:
        logger.debug("Qdrant entity payload filter failed (%s) — falling back to vector", exc)

    # Fallback: vector search on entity names
    return _vector_search_qdrant(" ".join(entities[:10]), limit)


def _build_bm25_index(chunks: list[dict[str, Any]]) -> None:
    """Build/rebuild the module-level BM25 index from a list of chunk dicts.

    Must be called explicitly — e.g. after ingestion — before _bm25_search()
    returns any results. Safe to call multiple times; replaces the previous index.

    Args:
        chunks: List of dicts with at least 'text' and optionally 'chunk_id' keys.

    Note:
        rank-bm25 is an optional dependency. If not installed this function is a
        no-op and _bm25_search() will continue to return [].
    """
    global _bm25_index, _bm25_corpus, _bm25_chunk_ids
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not installed — BM25 retrieval disabled. pip install rank-bm25")
        return

    tokenized = [c["text"].lower().split() for c in chunks]
    _bm25_corpus = [c["text"] for c in chunks]
    _bm25_chunk_ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
    _bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index built: %d documents", len(chunks))

    # Persist to disk so the index survives server restarts
    try:
        import pickle
        _BM25_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_BM25_PICKLE_PATH, "wb") as fh:
            pickle.dump(
                {"index": _bm25_index, "corpus": _bm25_corpus, "chunk_ids": _bm25_chunk_ids},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("BM25 index persisted to disk: %s", _BM25_PICKLE_PATH)
    except Exception as exc:
        logger.warning("BM25 disk save failed (non-fatal): %s", exc)


def _bm25_search(query: str, n: int = 20) -> list[dict[str, Any]]:
    """BM25 sparse retrieval over the module-level index.

    Excels at rare clinical terms (gene symbols, drug names, ICD codes) that
    dense embeddings dilute by averaging across semantic space.

    Returns [] gracefully when:
      - rank-bm25 is not installed
      - _build_bm25_index() has not been called yet
      - All BM25 scores are zero (query has no token overlap with corpus)

    Args:
        query: Raw or expanded query string.
        n:     Maximum results to return.

    Returns:
        List of chunk dicts sorted by BM25 score descending, with keys:
        chunk_id, text, score, source, metadata, retrieval_method.
    """
    global _bm25_index
    _load_bm25_from_disk()  # no-op after first call
    if _bm25_index is None:
        logger.debug("BM25 index not loaded — sparse retrieval disabled")
        return []

    try:
        import numpy as np

        tokenized_query = query.lower().split()
        scores = _bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:n]
        results: list[dict[str, Any]] = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "chunk_id": _bm25_chunk_ids[idx],
                    "text": _bm25_corpus[idx],
                    "score": float(scores[idx]),
                    "source": "",
                    "metadata": {},
                    "retrieval_method": "bm25",
                })
        logger.debug("BM25 search: %d results for query len=%d", len(results), len(tokenized_query))
        return results
    except Exception as exc:
        logger.error("BM25 search failed: %s", exc)
        return []


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for a document d across lists: sum(1 / (k + rank(d, list)))
    Higher score = more consistently highly ranked across lists.
    k=60 is the standard value from Cormack et al. (2009) and used by
    production systems including Elasticsearch's hybrid search.

    Args:
        ranked_lists: Each inner list is a ranked result list (index 0 = best).
                      Chunks are identified by their 'chunk_id' key; falls back
                      to first 50 chars of 'text' when chunk_id is absent.
        k:            RRF smoothing constant. Higher k reduces the influence of
                      top-rank positions. Default 60 is widely validated.

    Returns:
        Single merged list sorted by RRF score descending, preserving the
        chunk dict from whichever list last contributed that chunk_id.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            cid = chunk.get("chunk_id") or chunk.get("text", "")[:50]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]


def _expand_query_with_synonyms(query: str) -> str:
    """Add biomedical synonyms to query for better recall.

    Fast-path synonym expansion that runs regardless of graph availability.
    Covers the most common Alzheimer/stroke query terms with their canonical
    forms and abbreviations.  At most one synonym group is appended to avoid
    query-length bloat that degrades embedding quality.

    Args:
        query: Raw user query string.

    Returns:
        Query with one synonym group appended (or original if no match).
    """
    _SYNONYMS: dict[str, list[str]] = {
        # Alzheimer's disease cluster
        "alzheimer": ["alzheimer's disease", "AD", "senile dementia", "ADRD"],
        "memory loss": ["amnesia", "cognitive decline", "cognitive impairment", "dementia"],
        "amyloid": ["amyloid-beta", "Aβ", "beta-amyloid", "amyloid plaques", "Abeta"],
        "tau": ["tau protein", "neurofibrillary tangles", "NFT", "tauopathy"],
        "apoe": ["APOE4", "apolipoprotein E", "APOE epsilon4", "apoE"],
        "apoe4": ["APOE ε4", "apolipoprotein E4", "APOE epsilon4"],
        "neurodegeneration": ["neurodegeneration", "synaptic loss", "neuronal death"],
        "lecanemab": ["anti-amyloid antibody", "amyloid immunotherapy", "leqembi"],
        "aducanumab": ["anti-amyloid immunotherapy", "biogen antibody", "aduhelm"],
        # Stroke abbreviations
        "stroke": ["ischemic stroke", "cerebrovascular accident", "CVA", "brain infarction"],
        "ais": ["acute ischemic stroke", "ischaemic stroke", "acute stroke"],
        "tia": ["transient ischemic attack", "mini-stroke", "transient cerebral ischemia"],
        "ivt": ["intravenous thrombolysis", "IV alteplase", "tPA treatment"],
        "evt": ["endovascular thrombectomy", "mechanical thrombectomy", "thrombectomy"],
        "tpa": ["tissue plasminogen activator", "thrombolysis", "alteplase", "rt-PA"],
        "mcta": ["multiphase CTA", "multiphase computed tomography angiography", "CTA collaterals"],
        "cta": ["computed tomography angiography", "CT angiography", "vascular imaging"],
        "mrs": ["modified Rankin Scale", "functional outcome", "disability scale"],
        "nihss": ["NIH Stroke Scale", "stroke severity", "neurological deficit score"],
        "map": ["mean arterial pressure", "blood pressure", "arterial pressure"],
        "pp": ["pulse pressure", "systolic-diastolic difference"],
        # Pharmacology / treatment
        "donepezil": ["acetylcholinesterase inhibitor", "AChEI", "cholinesterase inhibitor"],
        "memantine": ["NMDA receptor antagonist", "glutamate antagonist"],
        "anti-hypertensive": ["antihypertensive", "blood pressure lowering", "hypertension treatment"],
        "inflammation": ["neuroinflammation", "microglia", "cytokine", "TNF-alpha"],
        "neuroprotection": ["neuroprotective", "brain protection", "ischemic preconditioning"],
        # Biomarkers / diagnostics
        "blood brain barrier": ["BBB", "blood-brain barrier", "cerebrovascular"],
        "csf": ["cerebrospinal fluid", "lumbar puncture", "CSF biomarker"],
        "pet": ["PET imaging", "positron emission tomography", "amyloid PET", "FDG-PET"],
        "mri": ["magnetic resonance imaging", "brain MRI", "neuroimaging"],
        "cognitive": ["cognition", "cognitive function", "executive function", "memory"],
        "dementia": ["neurodegenerative disease", "cognitive decline", "vascular dementia"],
        "biomarker": ["cerebrospinal fluid", "CSF biomarker", "plasma biomarker", "PET imaging"],
        # Clinical trial / study design terms
        "randomized": ["randomised controlled trial", "RCT", "clinical trial"],
        "cohort": ["prospective study", "observational study", "longitudinal study"],
        "meta-analysis": ["systematic review", "pooled analysis", "evidence synthesis"],
        "prevalence": ["incidence", "epidemiology", "disease frequency"],
        "odds ratio": ["OR", "relative risk", "hazard ratio", "risk estimate"],
        "confidence interval": ["CI", "95% CI", "statistical confidence"],
        "regression": ["logistic regression", "Cox regression", "multivariate analysis"],
        # Genetics
        "gwas": ["genome-wide association study", "genetic variant", "SNP"],
        "mutation": ["genetic variant", "polymorphism", "gene variant"],
    }
    query_lower = query.lower()
    for term, synonyms in _SYNONYMS.items():
        if term.lower() in query_lower:
            # Append first synonym only to avoid query-length bloat
            first_syn = synonyms[0]
            if first_syn.lower() not in query_lower:
                expanded = query + " " + first_syn
                logger.debug("Synonym expansion (fast-path): '%s' → +%r", term, first_syn)
                return expanded
            break
    return query


def _expand_query(query: str) -> str:
    """Expand query with ontology synonyms from the loaded entity graph.

    Runs the fast-path biomedical synonym expansion first, then augments
    further using the ontology graph's HPO/MONDO synonym map (up to 5
    additional canonical terms).

    Example:
        'memory loss' → 'memory loss amnesia cognitive impairment'

    Degrades gracefully when:
      - The graph file does not exist (returns synonym-expanded query)
      - pronto is not installed (build_synonym_map returns {} for ontology nodes)
      - The graph has no nodes (returns synonym-expanded query)

    Args:
        query: Raw user query string.

    Returns:
        Expanded query string (at minimum synonym-expanded).
    """
    # Fast-path: biomedical synonym expansion (graph-independent)
    query = _expand_query_with_synonyms(query)

    try:
        from mao.rag.ontology_loader import build_synonym_map

        G = load_graph()
        if G.number_of_nodes() == 0:
            return query

        synonym_map = build_synonym_map(G)
        if not synonym_map:
            return query

        extra_terms: list[str] = []
        query_lower = query.lower()
        for synonym, canonical in synonym_map.items():
            if synonym in query_lower and canonical.lower() not in query_lower:
                extra_terms.append(canonical)
                if len(extra_terms) >= 5:
                    break

        if extra_terms:
            expanded = query + " " + " ".join(extra_terms)
            logger.debug(
                "Query expanded: %d ontology synonym(s) added — '%s'",
                len(extra_terms),
                expanded[:120],
            )
            return expanded

        return query
    except Exception as exc:
        logger.debug("Ontology query expansion failed (%s) — using synonym-expanded query", exc)
        return query


def _mmr_dedup(chunks: list[Any], threshold: float = 0.85) -> list[Any]:
    """Remove near-duplicate chunks using token-overlap similarity.

    Applies a Maximum Marginal Relevance-style deduplication pass over an
    already-ranked list.  Iterates in score order (highest first) and drops
    any chunk whose Jaccard token overlap with an already-kept chunk exceeds
    ``threshold``.

    Only the first 300 characters of each chunk's text are compared, so
    two chunks that share a long common prefix but diverge significantly are
    kept.  This matches the most common form of duplication seen when the
    same Wikipedia paragraph appears in both the vector and BM25 result sets.

    Args:
        chunks:    Ranked list of chunk dicts or RankedChunk objects (index 0 = best score).
                   Each element must have either a 'text' attribute or 'text'/'content' key.
        threshold: Jaccard similarity threshold above which a chunk is
                   considered a near-duplicate.  Default 0.85 removes only
                   very close paraphrases while preserving topical diversity.

    Returns:
        Deduplicated list preserving order (highest-scoring chunk from each
        duplicate cluster is always kept).
    """
    if len(chunks) <= 1:
        return chunks

    seen_texts: list[str] = []
    kept: list[Any] = []

    for chunk in chunks:
        if hasattr(chunk, "text"):
            text = chunk.text[:300]
        else:
            text = chunk.get("text", chunk.get("content", ""))[:300]
        words = set(text.lower().split())
        is_dup = any(
            len(words & set(s.lower().split())) / max(len(words | set(s.lower().split())), 1) > threshold
            for s in seen_texts
        )
        if not is_dup:
            kept.append(chunk)
            seen_texts.append(text)

    dropped = len(chunks) - len(kept)
    if dropped:
        logger.debug("MMR dedup: removed %d near-duplicate chunk(s), %d kept", dropped, len(kept))
    return kept


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
