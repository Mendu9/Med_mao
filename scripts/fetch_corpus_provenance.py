"""Re-fetch authoritative provenance for the Corpus V1 PMC id set (P2-0 / A3).

Reads the PMC id list from ``mao/data/pmc_manifest.json`` and resolves real
bibliographic and license metadata from NCBI efetch, replacing the empty
fallback values the P2-0 audit found.

Usage:
    python scripts/fetch_corpus_provenance.py [--limit N] [--out PATH]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mao.corpus.licensing import decide  # noqa: E402
from mao.corpus.ncbi import fetch_provenance  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "mao" / "data" / "pmc_manifest.json"
DEFAULT_OUT = REPO / "build" / "corpus_v1" / "provenance.json"

logger = logging.getLogger("fetch_provenance")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="only fetch the first N ids")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pmcids = [r["pmcid"] for r in records if r.get("pmcid")]
    if args.limit:
        pmcids = pmcids[: args.limit]
    logger.info("resolving provenance for %d PMC ids", len(pmcids))

    def progress(done: int, total: int) -> None:
        if done % 200 == 0 or done >= total:
            logger.info("  %d/%d", done, total)

    result = fetch_provenance(
        pmcids,
        api_key=os.getenv("NCBI_API_KEY", "").strip(),
        # NCBI asks callers to identify themselves. Read it from the environment
        # rather than hardcoding a personal address into published source.
        email=os.getenv("PUBMED_EMAIL", "").strip(),
        progress=progress,
    )

    payload = {
        "requested": len(pmcids),
        "resolved": result.resolved,
        "missing": list(result.missing),
        "documents": {
            pmcid: {
                **dataclasses.asdict(doc),
                "authors": list(doc.authors),
                "pmc_url": doc.pmc_url,
                "redistributable": decide(doc.license_id).redistributable,
                "redistribution_reason": decide(doc.license_id).reason,
            }
            for pmcid, doc in sorted(result.documents.items())
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("resolved %d/%d -> %s", result.resolved, len(pmcids), args.out)
    if result.missing:
        logger.warning("%d ids unresolved", len(result.missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
