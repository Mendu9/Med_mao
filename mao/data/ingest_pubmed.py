"""
mao/data/ingest_pubmed.py
-------------------------
Ingests PubMed abstracts for Alzheimer and stroke research into ChromaDB.
Uses Bio.Entrez (biopython, free, no API key needed for < 3 req/s).
Adds abstracts to the same ChromaDB collection used by the retriever.
Also updates the entity graph (scispaCy NER) and BM25 index after ingestion.

Usage:
    python -m mao.data.ingest_pubmed
    python -m mao.data.ingest_pubmed --max-results 200

Import:
    from mao.data.ingest_pubmed import ingest_pubmed_abstracts
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PubMed search queries — spec-defined
# ---------------------------------------------------------------------------

_QUERIES: list[str] = [
    "Alzheimer disease treatment 2022:2025[dp]",
    "ischemic stroke therapy 2022:2025[dp]",
    "amyloid beta tau protein neurodegeneration 2023:2025[dp]",
]

_MAX_PER_QUERY = 300  # fetch up to 300 per query (900 total max)

# ---------------------------------------------------------------------------
# Biopython Entrez guard — clear error if not installed
# ---------------------------------------------------------------------------

try:
    from Bio import Entrez as _Entrez
    _BIOPYTHON_AVAILABLE = True
except ImportError:
    _BIOPYTHON_AVAILABLE = False
    _Entrez = None  # type: ignore[assignment]


def _require_biopython() -> None:
    if not _BIOPYTHON_AVAILABLE:
        raise ImportError(
            "biopython is required for PubMed ingestion. "
            "Install it with: pip install biopython"
        )


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _search_pubmed(query: str, max_results: int) -> list[str]:
    """Run Entrez esearch and return a PMID list."""
    _require_biopython()
    from Bio import Entrez

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance",
    )
    search_results = Entrez.read(handle)
    handle.close()
    time.sleep(0.4)  # stay under 3 req/s NCBI rate limit

    pmids: list[str] = search_results.get("IdList", [])
    logger.info("Query '%s' → %d PMIDs", query[:70], len(pmids))
    return pmids


def _fetch_pubmed_batch(pmids: list[str]) -> list[dict[str, Any]]:
    """Fetch and parse a batch of PubMed articles by PMID list.

    Returns list of {pmid, title, abstract, year, journal, source} dicts.
    Caller must sleep between batches if needed (this function sleeps 0.4 s).
    """
    _require_biopython()
    from Bio import Entrez

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    )
    records = Entrez.read(handle)
    handle.close()
    time.sleep(0.4)  # rate limit

    articles: list[dict[str, Any]] = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            art = medline["Article"]
            pmid = str(medline["PMID"])
            title = str(art.get("ArticleTitle", "")).strip()
            abstract_texts = art.get("Abstract", {}).get("AbstractText", [])
            abstract = (
                " ".join(str(t) for t in abstract_texts).strip()
                if abstract_texts
                else ""
            )

            if not abstract:
                continue

            pub_date = (
                art.get("Journal", {})
                .get("JournalIssue", {})
                .get("PubDate", {})
            )
            year = str(pub_date.get("Year", "")).strip()
            journal = str(art.get("Journal", {}).get("Title", "")).strip()

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "source": f"pubmed:{pmid}",
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to parse PubMed article: %s", exc)

    return articles


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_pubmed_abstracts(
    queries: list[str] | None = None,
    max_per_query: int = _MAX_PER_QUERY,
) -> int:
    """Fetch PubMed abstracts and ingest into ChromaDB + entity graph + BM25.

    This is the primary callable entry point for both CLI and API use.

    Args:
        queries: PubMed search strings. Defaults to the three spec-defined queries.
        max_per_query: Maximum abstracts fetched per query (default 300).

    Returns:
        Total number of chunks upserted to ChromaDB.
    """
    _require_biopython()
    from Bio import Entrez

    # Set Entrez email — NCBI requires a valid e-mail for API access
    pubmed_email = os.getenv("PUBMED_EMAIL")
    if not pubmed_email:
        raise ValueError("PUBMED_EMAIL env var must be set — NCBI requires a valid contact email for API access")
    Entrez.email = pubmed_email

    import hashlib

    import chromadb
    from chromadb.config import Settings

    from mao.core.config import cfg
    from mao.rag.embedder import embed_texts
    from mao.rag.chunker import adaptive_biomedical_chunk
    from mao.rag.graph_builder import (
        build_graph_from_documents,
        extract_entities,
        load_graph,
        save_graph,
    )

    if queries is None:
        queries = _QUERIES

    # Connect to ChromaDB — same collection as the rest of the corpus
    client = chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=cfg.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    # Load existing entity graph to extend (not replace) it
    graph = load_graph()

    all_docs: list[dict[str, Any]] = []
    total_chunks = 0

    for query in queries:
        pmids = _search_pubmed(query, max_results=max_per_query)
        if not pmids:
            continue

        # Fetch in batches of 50 to respect NCBI limits
        for batch_start in range(0, len(pmids), 50):
            batch = pmids[batch_start : batch_start + 50]
            try:
                articles = _fetch_pubmed_batch(batch)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Batch fetch failed (query='%s', offset=%d): %s",
                    query[:50], batch_start, exc,
                )
                continue

            for article in articles:
                doc_id = article["source"]
                full_text = f"{article['title']}\n\n{article['abstract']}"

                # Adaptive semantic chunking; falls back to fixed-size if
                # sentence-transformers is unavailable
                raw_chunks = adaptive_biomedical_chunk(full_text)
                if not raw_chunks:
                    continue

                chunk_dicts: list[dict[str, Any]] = []
                for idx, chunk_text in enumerate(raw_chunks):
                    chunk_id = hashlib.md5(
                        f"{doc_id}:{idx}:{chunk_text[:50]}".encode()
                    ).hexdigest()
                    chunk_dicts.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "source": doc_id,
                        "title": article["title"],
                        "journal": article["journal"],
                        "year": article["year"],
                        "domain": "pubmed",
                        "doc_type": "pubmed_abstract",
                        "chunk_index": idx,
                        "word_count": len(chunk_text.split()),
                    })

                # Enrich each chunk with NER entity list for graph edges
                for chunk in chunk_dicts:
                    try:
                        ents = [e[0] for e in extract_entities(chunk["text"])]
                        chunk["entities"] = ",".join(ents[:20])
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Entity extraction failed for %s: %s", doc_id, exc)
                        chunk["entities"] = ""

                # Upsert to ChromaDB
                try:
                    ids = [c["chunk_id"] for c in chunk_dicts]
                    documents = [c["text"] for c in chunk_dicts]
                    metadatas = [
                        {
                            k: v
                            for k, v in c.items()
                            if k != "text" and isinstance(v, (str, int, float, bool))
                        }
                        for c in chunk_dicts
                    ]
                    embeddings = embed_texts(documents)
                    collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings,
                    )
                    total_chunks += len(chunk_dicts)
                    all_docs.extend(chunk_dicts)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("ChromaDB upsert failed for %s: %s", doc_id, exc)

        logger.info("After query '%s': %d total chunks ingested", query[:70], total_chunks)

    # ------------------------------------------------------------------
    # Update entity graph with all ingested PubMed documents
    # ------------------------------------------------------------------
    if all_docs:
        try:
            import networkx as nx
            new_graph = build_graph_from_documents(all_docs)
            # Ensure both graphs share the same type before composing
            if type(graph) is not type(new_graph):
                compat = nx.MultiDiGraph()
                compat.add_nodes_from(graph.nodes(data=True))
                compat.add_edges_from(
                    (u, v, d) for u, v, d in graph.edges(data=True)
                )
                graph = compat
            merged = nx.compose(graph, new_graph)
            save_graph(merged)
            logger.info(
                "Entity graph updated: %d nodes, %d edges",
                merged.number_of_nodes(),
                merged.number_of_edges(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Entity graph update failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Rebuild BM25 sparse index so the retriever can use it immediately
    # ------------------------------------------------------------------
    if all_docs:
        try:
            from mao.rag.retriever import _build_bm25_index
            _build_bm25_index(all_docs)
            logger.info("BM25 index rebuilt: %d chunks", len(all_docs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 index rebuild failed (non-fatal): %s", exc)

    logger.info("PubMed ingestion complete: %d total chunks", total_chunks)
    return total_chunks


# ---------------------------------------------------------------------------
# Legacy alias — keeps the existing /ingest/pubmed API endpoint working
# (mao/api/main.py calls ingest_pubmed(domain=..., max_results_per_query=...))
# ---------------------------------------------------------------------------

def ingest_pubmed(
    domain: str = "all",
    max_results_per_query: int = _MAX_PER_QUERY,
) -> int:
    """Legacy wrapper for the API endpoint. Delegates to ingest_pubmed_abstracts()."""
    return ingest_pubmed_abstracts(max_per_query=max_results_per_query)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PubMed abstracts into the MAO knowledge base"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=_MAX_PER_QUERY,
        help=f"Max abstracts per query (default: {_MAX_PER_QUERY})",
    )
    args = parser.parse_args()

    total = ingest_pubmed_abstracts(max_per_query=args.max_results)
    print(f"\nPubMed ingestion complete: {total} chunks upserted to ChromaDB.")
    print("Query example:")
    print('  curl -X POST http://localhost:8080/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "What are current treatments for amyloid-beta aggregation?", "user_id": "test"}\'')


def live_pubmed_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Fetch PubMed abstracts for *query* and return as RAG-compatible snippets.

    This is the fallback path for graphrag_agent when Qdrant returns insufficient results.
    Does NOT ingest into ChromaDB — returns snippets directly for in-context use.

    Returns:
        list of {"title", "abstract", "source", "year", "journal"} dicts.
        Empty list if biopython unavailable, NCBI unreachable, or any error.
    """
    if not _BIOPYTHON_AVAILABLE:
        logger.debug("live_pubmed_search skipped — biopython not installed")
        return []

    pubmed_email = os.getenv("PUBMED_EMAIL")
    if not pubmed_email:
        logger.debug("live_pubmed_search skipped — PUBMED_EMAIL not set")
        return []

    try:
        from Bio import Entrez
        Entrez.email = pubmed_email
        pmids = _search_pubmed(query, max_results=max_results)
        if not pmids:
            return []
        articles = _fetch_pubmed_batch(pmids[:max_results])
        logger.info("Live PubMed fallback: %d articles for query %r", len(articles), query[:60])
        return [
            {
                "title":    a["title"],
                "abstract": a["abstract"][:600],
                "source":   a["source"],
                "year":     a["year"],
                "journal":  a["journal"],
            }
            for a in articles
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("live_pubmed_search failed (non-fatal): %s", exc)
        return []


if __name__ == "__main__":
    main()
