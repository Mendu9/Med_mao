"""
mao/data/reingest_all.py
------------------------
Full re-ingestion pipeline. Run this when the embedding model changes.

Steps:
  1. Drop the existing ChromaDB collection (clears all old vectors)
  2. Run ingest_alzheimers (22 local PDFs)
  3. Run ingest_pubmed (abstracts via PubMed Entrez)
  4. Run ingest_wikipedia (Wikipedia articles)
  5. Run ingest_pmc_papers (downloaded PMC full-text, if manifest exists)

Usage:
    python -m mao.data.reingest_all
    python -m mao.data.reingest_all --skip-drop      # keep existing collection
    python -m mao.data.reingest_all --only alzheimers pubmed
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_VALID_STEPS = ["alzheimers", "pubmed", "wikipedia", "pmc"]


def drop_collection() -> None:
    """Drop the ChromaDB collection so fresh embeddings can be upserted."""
    import chromadb
    from chromadb.config import Settings
    from mao.core.config import cfg

    client = chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(cfg.chroma_collection)
        logger.info("Dropped ChromaDB collection '%s'", cfg.chroma_collection)
    except Exception as exc:
        logger.warning("Could not drop collection (may not exist yet): %s", exc)


def reingest_all(
    steps: list[str] | None = None,
    skip_drop: bool = False,
    max_pubmed: int = 300,
    skip_triples: bool = False,
) -> dict[str, int]:
    """Run full re-ingestion pipeline.

    Args:
        steps: Which ingesters to run. None = all.
        skip_drop: If True, don't drop the existing ChromaDB collection first.
        max_pubmed: Max abstracts per PubMed query.
        skip_triples: If True, skip Groq triple extraction (avoids 429 rate limits).

    Returns:
        Dict of step -> chunks ingested.
    """
    if steps is None:
        steps = list(_VALID_STEPS)

    results: dict[str, int] = {}

    if not skip_drop:
        drop_collection()

    if "alzheimers" in steps:
        logger.info("=== Step: ingest_alzheimers ===")
        try:
            from mao.data.ingest_alzheimers import ingest_alzheimers_pdfs
            n = ingest_alzheimers_pdfs(skip_triples=skip_triples)
            results["alzheimers"] = n
            logger.info("ingest_alzheimers: %d chunks", n)
        except Exception as exc:
            logger.error("ingest_alzheimers failed: %s", exc)
            results["alzheimers"] = 0

    if "pubmed" in steps:
        logger.info("=== Step: ingest_pubmed ===")
        try:
            from mao.data.ingest_pubmed import ingest_pubmed_abstracts
            n = ingest_pubmed_abstracts(max_per_query=max_pubmed)
            results["pubmed"] = n
            logger.info("ingest_pubmed: %d chunks", n)
        except Exception as exc:
            logger.error("ingest_pubmed failed: %s", exc)
            results["pubmed"] = 0

    if "wikipedia" in steps:
        logger.info("=== Step: ingest_wikipedia ===")
        try:
            from mao.data.ingest_wikipedia import ingest_wikipedia_topics
            n = ingest_wikipedia_topics()
            results["wikipedia"] = n
            logger.info("ingest_wikipedia: %d chunks", n)
        except Exception as exc:
            logger.error("ingest_wikipedia failed: %s", exc)
            results["wikipedia"] = 0

    if "pmc" in steps:
        logger.info("=== Step: ingest_pmc_papers ===")
        try:
            from mao.data.download_pmc import _MANIFEST_PATH, ingest_pmc_papers
            if _MANIFEST_PATH.exists():
                n = ingest_pmc_papers()
                results["pmc"] = n
                logger.info("ingest_pmc: %d chunks", n)
            else:
                logger.info(
                    "No PMC manifest found at %s -- skipping (run download_pmc first)",
                    _MANIFEST_PATH,
                )
                results["pmc"] = 0
        except Exception as exc:
            logger.error("ingest_pmc failed: %s", exc)
            results["pmc"] = 0

    total = sum(results.values())
    logger.info("Re-ingestion complete. Total chunks: %d | breakdown: %s", total, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Drop ChromaDB collection and re-ingest all documents "
            "with the current EMBED_MODEL"
        )
    )
    parser.add_argument(
        "--skip-drop", action="store_true",
        help="Do not drop the existing collection before re-ingesting",
    )
    parser.add_argument(
        "--only", nargs="+", choices=_VALID_STEPS, default=None,
        help="Only run specific ingest steps",
    )
    parser.add_argument(
        "--max-pubmed", type=int, default=300,
        help="Max abstracts per PubMed query (default: 300)",
    )
    parser.add_argument(
        "--skip-triples", action="store_true",
        help="Skip Groq triple extraction (avoids 429 rate limits on large runs)",
    )
    args = parser.parse_args()

    from mao.core.config import cfg
    print(f"\nRe-ingesting with model: {cfg.embed_model} (dim={cfg.embed_dim})")
    print(f"ChromaDB: {cfg.chroma_host}:{cfg.chroma_port} / collection: {cfg.chroma_collection}")
    if not args.skip_drop:
        print("WARNING: This will DELETE the existing ChromaDB collection and all vectors.")
        confirm = input("Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    results = reingest_all(
        steps=args.only,
        skip_drop=args.skip_drop,
        max_pubmed=args.max_pubmed,
        skip_triples=args.skip_triples,
    )

    print("\nRe-ingestion complete:")
    for step, n in results.items():
        print(f"  {step}: {n} chunks")
    print(f"  TOTAL: {sum(results.values())} chunks")


if __name__ == "__main__":
    main()
