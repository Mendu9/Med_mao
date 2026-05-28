"""
mao/data/ingest_wikipedia.py
-----------------------------
Wikipedia + SQuAD ingestion pipeline — populates ChromaDB and builds the entity graph.

Two data sources:
  1. Wikipedia articles via wikipedia-api
  2. SQuAD 2.0 via Hugging Face datasets

Ingestion pipeline per article:
  1. Fetch full Wikipedia page text
  2. Chunk into overlapping windows (512 chars, 50 char overlap)
  3. Embed each chunk with nomic-embed-text
  4. Store in ChromaDB with metadata (source, title, chunk_id)
  5. Run NER → add to NetworkX entity graph
  6. Save graph to disk

Chunking strategy:
  - Character-based sliding window (simple, reproducible)
  - TODO(phase-2): switch to sentence-boundary chunking with nltk for
    better semantic coherence

Libraries:
  - wikipedia-api      (pip install wikipedia-api)
  - datasets           (pip install datasets)  — for SQuAD
  - chromadb           (pip install chromadb)
  - tqdm               (pip install tqdm)

Integration points:
  - rag/embedder.py       embed_texts()
  - rag/graph_builder.py  build_graph_from_documents(), save_graph()
  - core/config.py        chroma_collection, data_dir
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import chromadb
import wikipediaapi
from chromadb.config import Settings
from tqdm import tqdm

from mao.core.config import cfg
from mao.rag.embedder import embed_texts
from mao.rag.graph_builder import build_graph_from_documents, extract_entities, load_graph, save_graph

logger = logging.getLogger(__name__)

_CHUNK_SIZE    = 512
_CHUNK_OVERLAP = 50

# NER labels that are NOT biomedical — exclude from chunk entity metadata
_NON_BIO_LABELS: frozenset[str] = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC",
    "DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT",
    "MONEY", "QUANTITY", "EVENT", "LANGUAGE", "LAW",
    "WORK_OF_ART", "PRODUCT",
})

# ---------------------------------------------------------------------------
# Optional adaptive chunker — degrade gracefully if unavailable
# ---------------------------------------------------------------------------

try:
    from mao.rag.chunker import adaptive_biomedical_chunk as _adaptive_chunk
    _ADAPTIVE_CHUNKING_AVAILABLE = True
except Exception:  # noqa: BLE001
    _ADAPTIVE_CHUNKING_AVAILABLE = False
    logger.warning("mao.rag.chunker unavailable — falling back to fixed-size chunking")

_WIKI = wikipediaapi.Wikipedia(
    language="en",
    user_agent="MAO-Ingest/1.0",
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ingest_wikipedia_topics(
    topics: list[str],
    max_articles: int = 50,
) -> int:
    """
    Ingest a list of Wikipedia topic names into ChromaDB and the entity graph.

    Args:
        topics:       List of Wikipedia article titles (exact match).
        max_articles: Hard cap — prevent runaway ingestion.

    Returns:
        Number of articles successfully ingested.
    """
    collection = _get_chroma_collection()
    graph = load_graph()  # Load existing graph to extend it

    ingested = 0
    all_docs: list[dict[str, Any]] = []

    for topic in tqdm(topics[:max_articles], desc="Ingesting Wikipedia"):
        page = _WIKI.page(topic)
        if not page.exists():
            logger.warning("Wikipedia page not found: '%s'", topic)
            continue

        if _ADAPTIVE_CHUNKING_AVAILABLE:
            raw_chunks = _adaptive_chunk(page.text)
            chunks = []
            for idx, chunk_text in enumerate(raw_chunks):
                chunk_id = hashlib.md5(f"{topic}_{idx}".encode()).hexdigest()[:12]
                chunks.append({
                    "text": chunk_text,
                    "source": topic,
                    "title": page.title,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                })
        else:
            chunks = _chunk_text(page.text, source=topic, title=page.title)
        if not chunks:
            continue

        # Enrich each chunk with NER entities (biomedical labels only)
        for chunk in chunks:
            try:
                raw_ents = extract_entities(chunk["text"])
                bio_ents = [e for e, lbl in raw_ents if lbl not in _NON_BIO_LABELS]
                chunk["entities"] = ",".join(bio_ents[:20])
            except Exception:  # noqa: BLE001
                chunk["entities"] = ""

        # Store in ChromaDB
        _upsert_chunks(collection, chunks)
        all_docs.extend(chunks)
        ingested += 1
        logger.info("Ingested: %s (%d chunks)", topic, len(chunks))

    # Update entity graph with all newly ingested docs
    if all_docs:
        new_graph = build_graph_from_documents(all_docs)
        # Merge with existing graph
        merged = _merge_graphs(graph, new_graph)
        save_graph(merged)
        logger.info(
            "Entity graph updated: %d nodes, %d edges",
            merged.number_of_nodes(),
            merged.number_of_edges(),
        )

    return ingested


def ingest_squad(max_examples: int = 500) -> int:
    """
    Ingest SQuAD 2.0 context passages into ChromaDB.

    SQuAD contexts are short (1-3 paragraph) passages — ideal for
    demonstrating the retrieval pipeline on diverse topics.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: pip install datasets")
        return 0

    logger.info("Loading SQuAD 2.0 dataset...")
    ds = load_dataset("squad_v2", split="train")
    collection = _get_chroma_collection()

    # Deduplicate by context text (SQuAD has repeated contexts)
    seen_contexts: set[str] = set()
    docs: list[dict[str, Any]] = []

    for example in ds:
        context: str = example["context"]
        ctx_hash = hashlib.md5(context.encode()).hexdigest()[:12]
        if ctx_hash in seen_contexts:
            continue
        seen_contexts.add(ctx_hash)

        try:
            raw_ents_sq = extract_entities(context)
            squad_ents = ",".join(
                e for e, lbl in raw_ents_sq if lbl not in _NON_BIO_LABELS
            )
        except Exception:  # noqa: BLE001
            squad_ents = ""
        docs.append({
            "text": context,
            "source": "squad_v2",
            "chunk_id": f"squad_{ctx_hash}",
            "title": example.get("title", ""),
            "entities": squad_ents,
        })

        if len(docs) >= max_examples:
            break

    _upsert_chunks(collection, docs)
    logger.info("SQuAD ingestion complete: %d contexts", len(docs))
    return len(docs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    source: str,
    title: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{hashlib.md5(f'{source}_{idx}'.encode()).hexdigest()[:12]}"
            chunks.append({
                "text": chunk_text,
                "source": source,
                "title": title,
                "chunk_id": chunk_id,
                "chunk_index": idx,
            })
        start = end - overlap
        idx += 1
        if start >= len(text):
            break
    return chunks


def _upsert_chunks(
    collection: chromadb.Collection,
    chunks: list[dict[str, Any]],
) -> None:
    """Embed and upsert chunks into ChromaDB in batches."""
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts      = [c["text"] for c in batch]
        ids        = [c["chunk_id"] for c in batch]
        metadatas  = [
            {
                "source": c.get("source", ""),
                "title": c.get("title", ""),
                "chunk_index": str(c.get("chunk_index", 0)),
                "chunk_id": c.get("chunk_id", ""),
                "entities": c.get("entities", ""),
            }
            for c in batch
        ]

        try:
            embeddings = embed_texts(texts)
            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ChromaDB upsert failed for batch %d: %s", i, exc)


def _get_chroma_collection() -> chromadb.Collection:
    client = chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=cfg.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def _merge_graphs(g1, g2):
    """Merge two NetworkX graphs by combining nodes and edges."""
    import networkx as nx
    merged = nx.compose(g1, g2)
    # Sum co-occurrence weights for shared edges
    for u, v in g1.edges():
        if g2.has_edge(u, v):
            merged.edges[u, v]["weight"] = (
                g1.edges[u, v].get("weight", 1) + g2.edges[u, v].get("weight", 1)
            )
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Ingest data into MAO knowledge base")
    parser.add_argument("--topics", nargs="+", default=[
        "Machine learning",
        "Natural language processing",
        "Knowledge graph",
        "Transformer (deep learning)",
        "BERT (language model)",
        "Retrieval-augmented generation",
        "Graph database",
        "Attention mechanism",
        "Large language model",
        "Semantic search",
    ])
    parser.add_argument("--squad", action="store_true", help="Also ingest SQuAD 2.0")
    parser.add_argument("--max", type=int, default=20)
    args = parser.parse_args()

    n = ingest_wikipedia_topics(args.topics, max_articles=args.max)
    print(f"Wikipedia: {n} articles ingested")

    if args.squad:
        n2 = ingest_squad(max_examples=500)
        print(f"SQuAD: {n2} contexts ingested")
