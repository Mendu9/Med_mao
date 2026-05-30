"""
mao/data/download_pmc.py
------------------------
Bulk PMC Open Access full-text downloader.

Downloads full-text research papers from PubMed Central Open Access subset,
extracts body text from XML, saves as .txt files, and writes a manifest JSON.

Targets 200 papers per category (stroke, alzheimers, general_health) = 600 total.
Deduplicates by PMCID > PMID > DOI > normalized title across all categories.

Usage:
    python -m mao.data.download_pmc
    python -m mao.data.download_pmc --max-per-category 100
    python -m mao.data.download_pmc --categories stroke alzheimers
    python -m mao.data.download_pmc --resume   # skip already-downloaded PMCIDs

Output:
    mao/rag/data/pmc/{category}/PMC{id}.txt   — plain-text paper bodies
    mao/data/pmc_manifest.json                — metadata for all downloaded papers

Requires:
    pip install biopython lxml requests
    NCBI_API_KEY in .env (optional but gives 10 req/s vs 3 req/s)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_OUTPUT_BASE = _PROJECT_ROOT / "mao" / "rag" / "data" / "pmc"
_MANIFEST_PATH = Path(__file__).parent / "pmc_manifest.json"

# ---------------------------------------------------------------------------
# NCBI rate-limit: 3 req/s free, 10 req/s with API key
# ---------------------------------------------------------------------------

_NCBI_API_KEY = os.getenv("NCBI_API_KEY", "").strip()
_SLEEP = 0.11 if _NCBI_API_KEY else 0.35  # seconds between requests


def _sleep_rate() -> None:
    time.sleep(_SLEEP)


# ---------------------------------------------------------------------------
# Biopython guard
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
            "biopython is required: pip install biopython"
        )


# ---------------------------------------------------------------------------
# Step 1 — Search PMC for PMCIDs matching a query
# ---------------------------------------------------------------------------

def _search_pmc(query: str, max_results: int, retstart: int = 0) -> list[str]:
    """Run esearch on PMC db and return PMCID list.

    Appends 2016:2026[pdat] (correct PMC date field) to every query.
    Note: pmc[sb] and [dp] are PubMed-only — never include in PMC queries.
    """
    _require_biopython()
    from Bio import Entrez

    # Append PMC-compatible date filter (not [dp] which is PubMed-only)
    full_term = f"({query}) AND 2016:2026[pdat]"

    kwargs: dict[str, Any] = dict(
        db="pmc",
        term=full_term,
        retmax=max_results,
        retstart=retstart,
        sort="relevance",
    )
    if _NCBI_API_KEY:
        kwargs["api_key"] = _NCBI_API_KEY

    handle = Entrez.esearch(**kwargs)
    results = Entrez.read(handle)
    handle.close()
    _sleep_rate()

    pmcids: list[str] = [f"PMC{i}" for i in results.get("IdList", [])]
    logger.info("  esearch '%s...' -> %d PMCIDs", query[:60], len(pmcids))
    return pmcids


# ---------------------------------------------------------------------------
# Step 2 — Fetch article metadata (title, year, journal, doi, authors, license)
# ---------------------------------------------------------------------------

def _fetch_metadata_batch(pmcids: list[str]) -> list[dict[str, Any]]:
    """Fetch article metadata for up to 20 PMCIDs at once via efetch XML."""
    _require_biopython()
    from Bio import Entrez

    # Strip PMC prefix for the ID list — Entrez pmc db expects bare numbers
    ids = [p.replace("PMC", "") for p in pmcids]

    kwargs: dict[str, Any] = dict(
        db="pmc",
        id=",".join(ids),
        rettype="xml",
        retmode="xml",
    )
    if _NCBI_API_KEY:
        kwargs["api_key"] = _NCBI_API_KEY

    handle = Entrez.efetch(**kwargs)
    raw_xml = handle.read()
    handle.close()
    _sleep_rate()

    return _parse_metadata_xml(raw_xml, pmcids)


def _parse_metadata_xml(raw_xml: bytes, pmcids: list[str]) -> list[dict[str, Any]]:
    """Parse PMC efetch XML into metadata dicts."""
    try:
        from lxml import etree
    except ImportError:
        import xml.etree.ElementTree as etree  # type: ignore[no-redef]

    records: list[dict[str, Any]] = []
    try:
        root = etree.fromstring(raw_xml)
    except Exception as exc:
        logger.warning("XML parse failed: %s", exc)
        return records

    articles = root.findall(".//article")
    for art in articles:
        try:
            pmcid = _extract_text(art, ".//article-id[@pub-id-type='pmc']")
            if pmcid:
                pmcid = f"PMC{pmcid}"
            else:
                continue

            pmid = _extract_text(art, ".//article-id[@pub-id-type='pmid']") or ""
            doi = _extract_text(art, ".//article-id[@pub-id-type='doi']") or ""
            title = _extract_full_text(art, ".//article-title") or ""
            journal = _extract_text(art, ".//journal-title") or ""
            year = (
                _extract_text(art, ".//pub-date[@pub-type='epub']/year")
                or _extract_text(art, ".//pub-date/year")
                or ""
            )
            license_type = _extract_text(art, ".//license/@license-type") or "open-access"

            # Authors: last name + initials (first 5 then et al.)
            author_nodes = art.findall(".//contrib[@contrib-type='author']")
            authors = []
            for a in author_nodes[:5]:
                surname = _extract_text(a, ".//surname") or ""
                given = _extract_text(a, ".//given-names") or ""
                if surname:
                    authors.append(f"{surname} {given}".strip())
            if len(author_nodes) > 5:
                authors.append("et al.")

            records.append({
                "pmcid": pmcid,
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "journal": journal,
                "year": year,
                "authors": "; ".join(authors),
                "license": license_type,
                "pmc_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
            })
        except Exception as exc:
            logger.debug("Metadata parse error: %s", exc)

    return records


def _extract_text(node: Any, xpath: str) -> str:
    """Extract text from first matching element."""
    found = node.find(xpath)
    if found is None:
        return ""
    return (found.text or "").strip()


def _extract_full_text(node: Any, xpath: str) -> str:
    """Extract concatenated text+tail from all descendants (handles mixed content)."""
    found = node.find(xpath)
    if found is None:
        return ""
    try:
        from lxml import etree
        return "".join(found.itertext()).strip()
    except ImportError:
        parts = []
        for elem in found.iter():
            if elem.text:
                parts.append(elem.text)
            if elem.tail:
                parts.append(elem.tail)
        return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Step 3 — Download full-text XML and extract body text
# ---------------------------------------------------------------------------

def _download_fulltext(pmcid: str) -> str | None:
    """Download PMC full-text XML and extract clean body text.

    Uses the PMC OA Web Service XML endpoint (free, no auth required).
    Returns plain-text body or None if unavailable/restricted.
    """
    import urllib.request
    import urllib.error

    bare_id = pmcid.replace("PMC", "")
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={bare_id}&rettype=full&retmode=xml"
    )
    if _NCBI_API_KEY:
        url += f"&api_key={_NCBI_API_KEY}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw_xml = resp.read()
        _sleep_rate()
        return _extract_body_text(raw_xml)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            logger.debug("%s not in OA subset (HTTP %s)", pmcid, exc.code)
        else:
            logger.warning("HTTP %s downloading %s", exc.code, pmcid)
        _sleep_rate()
        return None
    except Exception as exc:
        logger.warning("Download failed for %s: %s", pmcid, exc)
        _sleep_rate()
        return None


# Section tags to extract — body first, then abstract
_SKIP_TAGS = {"ref-list", "ack", "app", "app-group", "supplementary-material", "fn-group"}


def _extract_body_text(raw_xml: bytes) -> str | None:
    """Extract clean body text from PMC XML, skipping refs and captions."""
    try:
        from lxml import etree
    except ImportError:
        import xml.etree.ElementTree as etree  # type: ignore[no-redef]

    try:
        root = etree.fromstring(raw_xml)
    except Exception:
        return None

    # Remove unwanted subtrees in-place
    for tag in _SKIP_TAGS:
        for node in root.findall(f".//{tag}"):
            parent = _find_parent(root, node)
            if parent is not None:
                try:
                    parent.remove(node)
                except Exception:
                    pass

    # Extract from <body> first, fall back to <abstract>
    body = root.find(".//body")
    if body is None:
        body = root.find(".//abstract")
    if body is None:
        return None

    skip_local = {"fig", "table-wrap", "disp-formula", "inline-formula"}
    parts: list[str] = []

    try:
        for elem in body.iter():
            if elem.tag in skip_local:
                continue
            if elem.text and elem.text.strip():
                parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                parts.append(elem.tail.strip())
    except Exception:
        pass

    text = " ".join(parts).strip()
    text = re.sub(r"\s+", " ", text)
    return text if len(text) > 200 else None


def _find_parent(root: Any, child: Any) -> Any:
    """Find parent of a node in an ElementTree (lxml or stdlib)."""
    try:
        return child.getparent()  # lxml
    except AttributeError:
        parent_map = {c: p for p in root.iter() for c in p}
        return parent_map.get(child)


# ---------------------------------------------------------------------------
# Main download pipeline
# ---------------------------------------------------------------------------

def download_pmc_papers(
    categories: list[str] | None = None,
    max_per_category: int = 200,
    resume: bool = True,
) -> dict[str, Any]:
    """Download full-text PMC papers for specified categories.

    Args:
        categories: Which SOURCE_GROUPS to process. None = all three.
        max_per_category: Target paper count per category (deduplicated).
        resume: Skip PMCIDs already present in the manifest.

    Returns:
        Summary dict: {total_downloaded, by_category, manifest_path, manifest_total}
    """
    _require_biopython()
    from Bio import Entrez
    from mao.data.queries import SOURCE_GROUPS, dedupe_records

    Entrez.email = os.getenv("PUBMED_EMAIL", "mendusaiarun9@gmail.com")

    if categories is None:
        categories = list(SOURCE_GROUPS.keys())

    # Load existing manifest for resume support
    existing_manifest: list[dict] = []
    if resume and _MANIFEST_PATH.exists():
        with open(_MANIFEST_PATH, encoding="utf-8") as f:
            existing_manifest = json.load(f)
    existing_pmcids = {r["pmcid"] for r in existing_manifest if r.get("pmcid")}
    logger.info("Manifest: %d existing records (resume=%s)", len(existing_pmcids), resume)

    all_new_records: list[dict] = []
    summary: dict[str, int] = {}

    for category in categories:
        if category not in SOURCE_GROUPS:
            logger.warning("Unknown category '%s', skipping", category)
            continue

        config = SOURCE_GROUPS[category]
        queries = config["queries"]
        logger.info("=== Category: %s | target: %d papers ===", category, max_per_category)

        category_pmcids: list[str] = []
        seen_in_category: set[str] = set()

        # Distribute search across all queries; fetch enough candidates to hit target after filtering
        per_query_target = max(10, (max_per_category * 2) // len(queries))

        for query in queries:
            if len(category_pmcids) >= max_per_category * 2:
                break
            pmcids = _search_pmc(query, max_results=per_query_target)
            for p in pmcids:
                if p not in seen_in_category:
                    seen_in_category.add(p)
                    category_pmcids.append(p)

        logger.info("Category %s: %d unique PMCIDs found", category, len(category_pmcids))

        # Filter out already-downloaded
        new_pmcids = [p for p in category_pmcids if p not in existing_pmcids]
        logger.info("Category %s: %d new (not yet downloaded)", category, len(new_pmcids))

        out_dir = _OUTPUT_BASE / category
        out_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        batch_size = 20

        for batch_start in range(0, len(new_pmcids), batch_size):
            if downloaded >= max_per_category:
                break

            batch = new_pmcids[batch_start : batch_start + batch_size]

            # Fetch metadata for this batch
            try:
                metadata_list = _fetch_metadata_batch(batch)
            except Exception as exc:
                logger.warning("Metadata fetch failed for batch at %d: %s", batch_start, exc)
                metadata_list = [{"pmcid": p} for p in batch]

            meta_by_pmcid = {m["pmcid"]: m for m in metadata_list}

            for pmcid in batch:
                if downloaded >= max_per_category:
                    break

                meta = meta_by_pmcid.get(pmcid, {"pmcid": pmcid})
                txt_path = out_dir / f"{pmcid}.txt"

                body_text = _download_fulltext(pmcid)
                if not body_text:
                    logger.debug("%s: no full text available, skipping", pmcid)
                    continue

                txt_path.write_text(body_text, encoding="utf-8")

                record: dict[str, Any] = {
                    "category": category,
                    "pmcid": pmcid,
                    "pmid": meta.get("pmid", ""),
                    "doi": meta.get("doi", ""),
                    "title": meta.get("title", pmcid),
                    "journal": meta.get("journal", ""),
                    "year": meta.get("year", ""),
                    "authors": meta.get("authors", ""),
                    "license": meta.get("license", "open-access"),
                    "pmc_url": meta.get("pmc_url", f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"),
                    "txt_path": str(txt_path.relative_to(_PROJECT_ROOT)),
                    "word_count": len(body_text.split()),
                    "query": "",
                }
                all_new_records.append(record)
                existing_pmcids.add(pmcid)
                downloaded += 1

                if downloaded % 10 == 0:
                    logger.info("  %s: %d/%d papers saved", category, downloaded, max_per_category)

        summary[category] = downloaded
        logger.info("Category %s complete: %d papers downloaded", category, downloaded)

    all_new_records = dedupe_records(all_new_records)

    updated_manifest = existing_manifest + all_new_records
    with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_manifest, f, indent=2, ensure_ascii=False)

    total = sum(summary.values())
    logger.info(
        "Download complete: %d new papers | manifest: %d total | saved to %s",
        total, len(updated_manifest), _MANIFEST_PATH,
    )

    return {
        "total_downloaded": total,
        "by_category": summary,
        "manifest_path": str(_MANIFEST_PATH),
        "manifest_total": len(updated_manifest),
    }


# ---------------------------------------------------------------------------
# Ingest downloaded papers into ChromaDB
# ---------------------------------------------------------------------------

def ingest_pmc_papers(
    categories: list[str] | None = None,
    manifest_path: Path | None = None,
) -> int:
    """Ingest downloaded PMC papers from manifest into ChromaDB.

    Reads each .txt file referenced in the manifest, runs adaptive chunking,
    builds entity graph, and upserts to ChromaDB.

    Returns: total chunks upserted.
    """
    import chromadb
    from chromadb.config import Settings

    from mao.core.config import cfg
    from mao.rag.embedder import embed_texts
    from mao.rag.chunker import chunk_document
    from mao.rag.graph_builder import (
        build_graph_from_documents,
        extract_entities,
        load_graph,
        save_graph,
    )

    mp = manifest_path or _MANIFEST_PATH
    if not mp.exists():
        raise FileNotFoundError(f"Manifest not found: {mp}. Run download_pmc_papers() first.")

    with open(mp, encoding="utf-8") as f:
        manifest: list[dict] = json.load(f)

    if categories:
        manifest = [r for r in manifest if r.get("category") in categories]

    client = chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=cfg.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    graph = load_graph()
    all_chunk_dicts: list[dict] = []
    total_chunks = 0

    for record in manifest:
        txt_path = _PROJECT_ROOT / record.get("txt_path", "")
        if not txt_path.exists():
            logger.debug("txt file missing: %s", txt_path)
            continue

        text = txt_path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        doc_id = f"pmc:{record['pmcid']}"
        category = record.get("category", "general")

        chunk_dicts = chunk_document(
            text=text,
            doc_id=doc_id,
            metadata={
                "source": doc_id,
                "title": record.get("title", ""),
                "journal": record.get("journal", ""),
                "year": record.get("year", ""),
                "authors": record.get("authors", ""),
                "doi": record.get("doi", ""),
                "pmc_url": record.get("pmc_url", ""),
                "domain": category,
                "doc_type": "pmc_fulltext",
                "license": record.get("license", "open-access"),
            },
        )

        for chunk in chunk_dicts:
            try:
                ents = [e[0] for e in extract_entities(chunk["text"])]
                chunk["entities"] = ",".join(ents[:20])
            except Exception:
                chunk["entities"] = ""

        if chunk_dicts:
            try:
                ids = [c["chunk_id"] for c in chunk_dicts]
                documents = [c["text"] for c in chunk_dicts]
                metadatas = [
                    {k: v for k, v in c.items() if k != "text" and isinstance(v, (str, int, float, bool))}
                    for c in chunk_dicts
                ]
                embeddings = embed_texts(documents)
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                total_chunks += len(chunk_dicts)
                all_chunk_dicts.extend(chunk_dicts)
            except Exception as exc:
                logger.warning("ChromaDB upsert failed for %s: %s", doc_id, exc)

    if all_chunk_dicts:
        try:
            import networkx as nx
            new_graph = build_graph_from_documents(all_chunk_dicts)
            if type(graph) is not type(new_graph):
                compat = nx.MultiDiGraph()
                compat.add_nodes_from(graph.nodes(data=True))
                compat.add_edges_from((u, v, d) for u, v, d in graph.edges(data=True))
                graph = compat
            merged = nx.compose(graph, new_graph)
            save_graph(merged)
            logger.info(
                "Entity graph updated: %d nodes, %d edges",
                merged.number_of_nodes(), merged.number_of_edges(),
            )
        except Exception as exc:
            logger.warning("Entity graph update failed (non-fatal): %s", exc)

        try:
            from mao.rag.retriever import _build_bm25_index
            _build_bm25_index(all_chunk_dicts)
            logger.info("BM25 index rebuilt: %d chunks", len(all_chunk_dicts))
        except Exception as exc:
            logger.warning("BM25 rebuild failed (non-fatal): %s", exc)

    logger.info("PMC ingestion complete: %d chunks upserted", total_chunks)
    return total_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PMC full-text papers and ingest into ChromaDB"
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        choices=["stroke", "alzheimers", "general_health"],
        help="Categories to download (default: all three)",
    )
    parser.add_argument(
        "--max-per-category", type=int, default=200,
        help="Target paper count per category (default: 200)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-download even if PMCID already in manifest",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download .txt files but do not ingest into ChromaDB",
    )
    parser.add_argument(
        "--ingest-only", action="store_true",
        help="Ingest from existing manifest without downloading",
    )
    args = parser.parse_args()

    if not args.ingest_only:
        result = download_pmc_papers(
            categories=args.categories,
            max_per_category=args.max_per_category,
            resume=not args.no_resume,
        )
        print(f"\nDownload complete:")
        print(f"  Total new papers: {result['total_downloaded']}")
        for cat, n in result["by_category"].items():
            print(f"  {cat}: {n}")
        print(f"  Manifest: {result['manifest_path']} ({result['manifest_total']} total records)")

    if not args.download_only:
        total_chunks = ingest_pmc_papers(categories=args.categories)
        print(f"\nIngestion complete: {total_chunks} chunks upserted to ChromaDB.")


if __name__ == "__main__":
    main()
