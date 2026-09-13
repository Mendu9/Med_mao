"""Build Corpus V1 — document-level, provenance-verified (P2-0 / A1+A2).

Inputs
    build/corpus_v1/provenance.json      authoritative NCBI metadata (A3)
    mao/rag/data/pmc/<category>/*.txt    tracked full-text sources

Outputs
    build/corpus_v1/documents.parquet    canonical document artifact
    build/corpus_v1/exclusions.json      every rejected document, with reason
    build/corpus_v1/corpus_manifest.json artifact manifest + checksums

The 42 legacy PDFs are excluded by construction: they are not PMC sources and
have no recoverable redistribution rights (P2-0 audit, decision A2).
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mao.corpus.build import build_corpus  # noqa: E402
from mao.corpus.provenance import DocumentProvenance  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build" / "corpus_v1"
PMC_TEXT_ROOT = REPO / "mao" / "rag" / "data" / "pmc"

CORPUS_VERSION = "1.0.0"
BUILDER_VERSION = "1.0.0"

logger = logging.getLogger("build_corpus_v1")


def _text_index() -> dict[str, Path]:
    return {path.stem: path for path in PMC_TEXT_ROOT.rglob("*.txt")}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    payload = json.loads((BUILD / "provenance.json").read_text(encoding="utf-8"))
    provenance = {
        pmcid: DocumentProvenance(
            pmcid=record["pmcid"],
            pmid=record["pmid"],
            doi=record["doi"],
            title=record["title"],
            journal=record["journal"],
            year=record["year"],
            authors=tuple(record["authors"]),
            license_url=record["license_url"],
            license_type=record["license_type"],
            license_content_type=record["license_content_type"],
            license_id=record["license_id"],
            copyright_statement=record["copyright_statement"],
            article_type=record["article_type"],
            is_retracted=record["is_retracted"],
        )
        for pmcid, record in payload["documents"].items()
    }
    logger.info("loaded provenance for %d documents", len(provenance))

    index = _text_index()
    logger.info("indexed %d full-text files under %s", len(index), PMC_TEXT_ROOT)

    def text_for(pmcid: str) -> str | None:
        path = index.get(pmcid)
        return path.read_text(encoding="utf-8", errors="replace") if path else None

    result = build_corpus(provenance, text_for=text_for)
    logger.info("admitted %d, excluded %d", result.admitted, len(result.excluded))

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [dataclasses.asdict(doc) for doc in result.documents]
    for row in rows:
        row["authors"] = list(row["authors"])
    table = pa.Table.from_pylist(rows)

    BUILD.mkdir(parents=True, exist_ok=True)
    parquet_path = BUILD / "documents.parquet"
    pq.write_table(table, parquet_path, compression="zstd")

    (BUILD / "exclusions.json").write_text(
        json.dumps([dataclasses.asdict(e) for e in result.excluded], indent=2),
        encoding="utf-8",
    )

    by_license = collections.Counter(d.license_id for d in result.documents)
    by_reason = collections.Counter(e.reason for e in result.excluded)
    manifest = {
        "artifact_id": "mao-corpus-source",
        "corpus_version": CORPUS_VERSION,
        "builder_version": BUILDER_VERSION,
        "granularity": "document",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "provider": "NCBI PMC (efetch)",
            "id_list": "mao/data/pmc_manifest.json",
            "text_root": str(PMC_TEXT_ROOT.relative_to(REPO)).replace("\\", "/"),
        },
        "counts": {
            "candidates": len(provenance),
            "admitted": result.admitted,
            "excluded": len(result.excluded),
            "excluded_by_reason": dict(by_reason),
            "legacy_pdfs_excluded": 42,
        },
        "license_summary": dict(by_license),
        "totals": {
            "words": sum(d.word_count for d in result.documents),
            "chars": sum(d.char_count for d in result.documents),
        },
        "files": [
            {
                "path": "documents.parquet",
                "sha256": _sha256_file(parquet_path),
                "bytes": parquet_path.stat().st_size,
                "rows": result.admitted,
            }
        ],
        "policy": {
            "redistribution": "fail-closed; only positively identified permissive licenses admitted",
            "no_derivatives": "excluded — a chunked corpus is a derivative work",
            "retracted": "excluded — not admissible as clinical evidence",
        },
    }
    (BUILD / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info("wrote %s (%.1f MB)", parquet_path, parquet_path.stat().st_size / 1e6)
    logger.info("exclusions by reason: %s", dict(by_reason))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
