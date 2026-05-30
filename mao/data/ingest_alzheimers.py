"""
mao/data/ingest_alzheimers.py
------------------------------
Ingests Alzheimer's disease research PDFs into ChromaDB + entity graph.

Replaces the naive RAG in ad/rag/ (FAISS + Groq + all-MiniLM-L6-v2) with
MAO's GraphRAG pipeline:
  - Embedder:   nomic-embed-text via Ollama (local, free)
  - Vector DB:  ChromaDB (same instance as Wikipedia knowledge base)
  - Graph:      NetworkX entity graph (same graph extended with AD entities)
  - Retrieval:  rag/retriever.py — 5-step GraphRAG pipeline at query time

Once ingested, any user question routed to graphrag_agent will automatically
retrieve from both Wikipedia chunks AND Alzheimer's PDF chunks. No router
changes needed.

Default PDF source: d:/project/ad/rag/data/ (22 research PDFs)
Each chunk stored with metadata: source=filename, domain=alzheimers

Usage:
    python -m mao.data.ingest_alzheimers
    python -m mao.data.ingest_alzheimers --data-dir path/to/pdfs
    python -m mao.data.ingest_alzheimers --data-dir path/to/pdfs --chunk-size 512

Libraries:
    pypdf   (pip install pypdf)
    tqdm    (pip install tqdm)
    chromadb (already a MAO dependency)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from mao.core.config import cfg
from mao.rag.embedder import embed_texts
from mao.rag.graph_builder import build_graph_from_documents, extract_entities, load_graph, save_graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Sprint 2B imports — degrade gracefully if not yet installed
# ---------------------------------------------------------------------------

try:
    from mao.rag.chunker import chunk_document as _chunk_document_fn
    _ADAPTIVE_CHUNKING_AVAILABLE = True
except Exception:  # noqa: BLE001
    _ADAPTIVE_CHUNKING_AVAILABLE = False
    logger.warning("mao.rag.chunker unavailable — falling back to fixed-size chunking")

try:
    from mao.rag.triple_extractor import add_triples_to_graph, extract_triples
    _TRIPLE_EXTRACTION_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TRIPLE_EXTRACTION_AVAILABLE = False
    logger.warning("mao.rag.triple_extractor unavailable — semantic triple edges skipped")

_CHUNK_SIZE    = 800   # ~2-3 paragraphs with complete sentences
_CHUNK_OVERLAP = 100   # preserves cross-boundary context

# Default path to the AD research PDFs — mao/rag/data/ (primary), AD/rag/data/ (legacy fallback)
_DEFAULT_PDF_DIR = Path(__file__).resolve().parents[1] / "rag" / "data"
_LEGACY_PDF_DIR  = Path(__file__).resolve().parents[2] / "AD" / "rag" / "data"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ingest_alzheimers_pdfs(
    data_dir: str | Path | None = None,
    chunk_size: int = _CHUNK_SIZE,
    chunk_overlap: int = _CHUNK_OVERLAP,
    skip_triples: bool = False,
) -> int:
    """
    Ingest all PDFs in *data_dir* into ChromaDB and extend the entity graph.

    Args:
        data_dir:     Directory containing .pdf files. Defaults to ad/rag/data/.
        chunk_size:   Characters per chunk (default 512).
        chunk_overlap: Overlap between consecutive chunks (default 50).
        skip_triples: If True, skip Groq triple extraction (avoids 429 on large runs).

    Returns:
        Total number of PDF files successfully ingested.
    """
    if data_dir:
        pdf_dir = Path(data_dir)
    elif _DEFAULT_PDF_DIR.exists():
        pdf_dir = _DEFAULT_PDF_DIR
    elif _LEGACY_PDF_DIR.exists():
        logger.info("Using legacy PDF dir: %s", _LEGACY_PDF_DIR)
        pdf_dir = _LEGACY_PDF_DIR
    else:
        pdf_dir = _DEFAULT_PDF_DIR  # will trigger error below

    if not pdf_dir.exists():
        logger.error("PDF directory not found: %s", pdf_dir)
        return 0

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in: %s", pdf_dir)
        return 0

    logger.info("Found %d PDF files in %s", len(pdf_files), pdf_dir)

    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return 0

    collection = _get_chroma_collection()
    graph = load_graph()  # Extend the existing entity graph

    ingested = 0
    all_docs: list[dict[str, Any]] = []

    for pdf_path in tqdm(pdf_files, desc="Ingesting Alzheimer's PDFs"):
        try:
            text = _extract_pdf_text(pdf_path, PdfReader)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", pdf_path.name, exc)
            continue

        if not text.strip():
            logger.warning("No extractable text in %s — skipping", pdf_path.name)
            continue

        source_name = pdf_path.name

        # ------------------------------------------------------------------
        # Chunking — adaptive (Sprint 2B) with fallback to fixed-size
        # ------------------------------------------------------------------
        if _ADAPTIVE_CHUNKING_AVAILABLE:
            chunks = _chunk_document_fn(
                text=text,
                doc_id=source_name,
                metadata={
                    "source": source_name,
                    "domain": "alzheimer",
                    "title": pdf_path.stem,
                    "doc_type": "research_paper",
                },
                use_sections=True,
            )
        else:
            chunks = _chunk_text(
                text,
                source=source_name,
                title=pdf_path.stem,
                domain="alzheimer",
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

        if not chunks:
            continue

        # ------------------------------------------------------------------
        # Enrich each chunk with comma-separated NER entities
        # ChromaDB requires metadata values to be str/int/float/bool — never list
        # ------------------------------------------------------------------
        for chunk in chunks:
            try:
                ents = [e[0] for e in extract_entities(chunk["text"])]
                chunk["entities"] = ",".join(ents[:20])
            except Exception as exc:  # noqa: BLE001
                logger.debug("Entity extraction failed for chunk %s: %s", chunk.get("chunk_id"), exc)
                chunk["entities"] = ""

        _upsert_chunks(collection, chunks)
        all_docs.extend(chunks)
        ingested += 1
        logger.info("Ingested: %s (%d chunks)", source_name, len(chunks))

    # Update entity graph with all AD documents
    if all_docs:
        new_graph = build_graph_from_documents(all_docs)
        import networkx as nx
        # Ensure both graphs are the same type before composing.
        # The saved graph may be a plain Graph/DiGraph from an older run.
        if type(graph) is not type(new_graph):
            compat = nx.MultiDiGraph()
            compat.add_nodes_from(graph.nodes(data=True))
            compat.add_edges_from((u, v, d) for u, v, d in graph.edges(data=True))
            graph = compat
        merged = nx.compose(graph, new_graph)

        # ------------------------------------------------------------------
        # Triple extraction (Sprint 2B) — adds typed semantic edges
        # (treats/causes/etc.) alongside co-occurrence edges from graph_builder.
        # Only first 5 chunks per PDF to limit Ollama call volume.
        # ------------------------------------------------------------------
        if _TRIPLE_EXTRACTION_AVAILABLE and not skip_triples:
            # Reconstruct per-PDF chunk groups from all_docs
            from itertools import groupby
            key_fn = lambda c: c.get("source", "")  # noqa: E731
            docs_by_source = {
                src: list(grp)
                for src, grp in groupby(
                    sorted(all_docs, key=key_fn), key=key_fn
                )
            }
            triple_count = 0
            for src, doc_chunks in docs_by_source.items():
                for chunk in doc_chunks[:5]:  # first 5 chunks per PDF
                    try:
                        triples = extract_triples(chunk["text"])
                        if triples:
                            added = add_triples_to_graph(merged, triples, doc_id=chunk.get("source", src))
                            triple_count += added
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Triple extraction failed for %s: %s", src, exc)
            logger.info("Semantic triples added to graph: %d edges", triple_count)

        save_graph(merged)
        logger.info(
            "Entity graph updated: %d nodes, %d edges",
            merged.number_of_nodes(),
            merged.number_of_edges(),
        )

    # Build BM25 sparse index from all ingested chunks so the retriever's
    # step 3 (_bm25_search) works immediately after ingestion.
    if all_docs:
        try:
            from mao.rag.retriever import _build_bm25_index
            _build_bm25_index(all_docs)
            logger.info("BM25 index built: %d chunks", len(all_docs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 index build failed (non-fatal): %s", exc)

    logger.info("Alzheimer's ingestion complete: %d/%d PDFs ingested", ingested, len(pdf_files))
    return ingested


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: Path, PdfReader) -> str:
    """
    Extract all text from a PDF using pypdf.

    Returns concatenated text from all pages with double-newline separation.
    Pages with no extractable text are silently skipped.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        except Exception:  # noqa: BLE001
            continue
    return "\n\n".join(pages)


def _recursive_split(
    text: str,
    size: int,
    overlap: int,
    seps: list[str] | None = None,
) -> list[str]:
    """
    Recursive character splitting — tries paragraph → line → sentence → word.
    Never cuts mid-sentence if a coarser separator fits.
    """
    if seps is None:
        seps = ["\n\n", "\n", ". ", " "]
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        results, buf = [], ""
        for part in parts:
            candidate = buf + (sep if buf else "") + part
            if len(candidate) <= size:
                buf = candidate
            elif buf:
                results.append(buf.strip())
                tail = buf[-overlap:] if len(buf) > overlap else buf
                buf = (tail + sep + part).strip()
            else:
                # Single part too large — recurse with finer separator
                idx = seps.index(sep)
                results.extend(_recursive_split(part, size, overlap, seps[idx + 1 :]))
                buf = ""
        if buf.strip():
            results.append(buf.strip())
        return [r for r in results if r]
    return [text[:size].strip()]


def _chunk_text(
    text: str,
    source: str,
    title: str,
    domain: str = "alzheimers",
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Recursively split text preserving paragraph/sentence boundaries."""
    raw_chunks = _recursive_split(text, chunk_size, overlap)
    result = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = hashlib.md5(f"{source}_{idx}".encode()).hexdigest()[:12]
        result.append({
            "text": chunk_text,
            "source": source,
            "title": title,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "domain": domain,
        })
    return result


def _upsert_chunks(collection: chromadb.Collection, chunks: list[dict]) -> None:
    """Embed and upsert chunks into ChromaDB one at a time to avoid OOM on large batches."""
    for i, chunk in enumerate(chunks):
        try:
            embedding = embed_texts([chunk["text"]])[0]
            metadata = {k: v for k, v in chunk.items() if k != "text"}
            # ChromaDB metadata values must be str/int/float/bool
            metadata = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                        for k, v in metadata.items()}
            collection.upsert(
                ids=[chunk["chunk_id"]],
                documents=[chunk["text"]],
                metadatas=[metadata],
                embeddings=[embedding],
            )
        except Exception as exc:
            logger.error("ChromaDB upsert failed for chunk %d (%s): %s", i, chunk.get("chunk_id"), exc)


def _get_chroma_collection() -> chromadb.Collection:
    """Return the shared MAO ChromaDB collection, creating it if needed."""
    client = chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=cfg.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Ingest Alzheimer's research PDFs into MAO knowledge base"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"Path to PDF directory (default: {_DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_CHUNK_SIZE,
        help="Characters per chunk (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=_CHUNK_OVERLAP,
        help="Overlap between chunks (default: 50)",
    )
    args = parser.parse_args()

    n = ingest_alzheimers_pdfs(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"\nIngestion complete: {n} PDF files processed into ChromaDB.")
    print("Query example:")
    print('  curl -X POST http://localhost:8080/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "What is the role of tau protein in Alzheimer disease?", "user_id": "test"}\'')
