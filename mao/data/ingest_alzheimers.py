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

from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

from mao.core.config import cfg
from mao.rag.embedder import embed_texts
from mao.rag.graph_builder import build_graph_from_documents, load_graph, save_graph

logger = logging.getLogger(__name__)

_CHUNK_SIZE    = 800   # ~2-3 paragraphs with complete sentences
_CHUNK_OVERLAP = 100   # preserves cross-boundary context

# Default path to the AD research PDFs (relative to project root)
_DEFAULT_PDF_DIR = Path(__file__).resolve().parents[3] / "ad" / "rag" / "data"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ingest_alzheimers_pdfs(
    data_dir: str | Path | None = None,
    chunk_size: int = _CHUNK_SIZE,
    chunk_overlap: int = _CHUNK_OVERLAP,
) -> int:
    """
    Ingest all PDFs in *data_dir* into ChromaDB and extend the entity graph.

    Args:
        data_dir:     Directory containing .pdf files. Defaults to ad/rag/data/.
        chunk_size:   Characters per chunk (default 512).
        chunk_overlap: Overlap between consecutive chunks (default 50).

    Returns:
        Total number of PDF files successfully ingested.
    """
    pdf_dir = Path(data_dir) if data_dir else _DEFAULT_PDF_DIR

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

    index = _get_pinecone_index()
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
        print("pdf_path", pdf_path)
        chunks = _chunk_text(
            text,
            source=pdf_path.name,
            title=pdf_path.stem,
            domain="alzheimers",
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        if not chunks:
            continue

        _upsert_chunks(index, chunks)
        all_docs.extend(chunks)
        ingested += 1
        logger.info("Ingested: %s (%d chunks)", pdf_path.name, len(chunks))

    # Update entity graph with all AD documents
    if all_docs:
        new_graph = build_graph_from_documents(all_docs)
        merged = _merge_graphs(graph, new_graph)
        save_graph(merged)
        logger.info(
            "Entity graph updated: %d nodes, %d edges",
            merged.number_of_nodes(),
            merged.number_of_edges(),
        )

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


def _upsert_chunks(index, chunks: list[dict]) -> None:
    """Embed and upsert chunks into Pinecone in batches of 64."""
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        texts = [c["text"] for c in batch]
        try:
            embeddings = embed_texts(texts)
            vectors = [
                {
                    "id": c["chunk_id"],
                    "values": emb,
                    "metadata": {
                        "text":        c.get("text", ""),
                        "source":      c.get("source", ""),
                        "title":       c.get("title", ""),
                        "chunk_index": c.get("chunk_index", 0),
                        "chunk_id":    c.get("chunk_id", ""),
                        "domain":      c.get("domain", "alzheimers"),
                    },
                }
                for c, emb in zip(batch, embeddings)
            ]
            index.upsert(vectors=vectors)
        except Exception as exc:
            logger.error("Pinecone upsert failed for batch %d: %s", i, exc)


def _get_pinecone_index():
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if cfg.pinecone_index not in existing:
        pc.create_index(
            name=cfg.pinecone_index,
            dimension=cfg.embed_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=cfg.pinecone_region),
        )
        logger.info("Created Pinecone index: %s", cfg.pinecone_index)
    return pc.Index(cfg.pinecone_index)


def _merge_graphs(g1, g2):
    """Merge two NetworkX graphs, summing co-occurrence weights on shared edges."""
    import networkx as nx
    merged = nx.compose(g1, g2)
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
