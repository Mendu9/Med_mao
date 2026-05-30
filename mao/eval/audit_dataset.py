"""
mao/eval/audit_dataset.py
--------------------------
Audit the golden dataset against the live ChromaDB collection.

Run:
    python -m mao.eval.audit_dataset
    python -m mao.eval.audit_dataset --dataset mao/data/golden_dataset.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from collections import Counter

import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION = "mao_knowledge"  # cfg.chroma_collection default

GOLDEN_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "golden_dataset.json"

# Heuristics for metadata/low-quality questions
_METADATA_PATTERNS = re.compile(
    r"\b(doi|journal|published|license|copyright|issn|pmid|pmcid|volume|issue|pages?)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def audit(dataset_path: Path = GOLDEN_DATASET_PATH) -> None:
    # ------------------------------------------------------------------
    # 1. Load golden dataset
    # ------------------------------------------------------------------
    logger.info("Loading golden dataset from %s", dataset_path)
    with dataset_path.open(encoding="utf-8") as fh:
        records: list[dict] = json.load(fh)

    total = len(records)
    logger.info("Total samples in golden dataset: %d", total)

    # ------------------------------------------------------------------
    # 2. Question quality checks
    # ------------------------------------------------------------------
    questions = [r["question"] for r in records]

    # Duplicate questions
    question_counts = Counter(questions)
    duplicates = {q: c for q, c in question_counts.items() if c > 1}
    if duplicates:
        logger.warning("DUPLICATE questions found (%d):", len(duplicates))
        for q, c in duplicates.items():
            logger.warning("  [x%d] %s", c, q)
    else:
        logger.info("No duplicate questions found.")

    # Metadata-like questions (mentions doi, journal, published, license, etc.)
    metadata_questions = [q for q in questions if _METADATA_PATTERNS.search(q)]
    if metadata_questions:
        logger.warning(
            "Questions that look like metadata (%d):", len(metadata_questions)
        )
        for q in metadata_questions:
            logger.warning("  [META] %s", q)
    else:
        logger.info("No metadata-like questions found.")

    # ------------------------------------------------------------------
    # 3. Connect to ChromaDB
    # ------------------------------------------------------------------
    logger.info(
        "Connecting to ChromaDB at %s:%s, collection=%s",
        CHROMA_HOST,
        CHROMA_PORT,
        CHROMA_COLLECTION,
    )
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection(name=CHROMA_COLLECTION)
    except Exception as exc:
        logger.error("Failed to get collection '%s': %s", CHROMA_COLLECTION, exc)
        logger.error("Is ChromaDB running? Try: docker compose up chromadb")
        return

    chroma_count = collection.count()
    logger.info("ChromaDB collection '%s' has %d documents.", CHROMA_COLLECTION, chroma_count)

    # ------------------------------------------------------------------
    # 4. For each entry check if relevant_chunk_ids[0] exists in ChromaDB
    # ------------------------------------------------------------------
    found = 0
    missing = 0
    text_match = 0
    text_mismatch = 0
    text_missing = 0  # chunk found but has no document text stored

    missing_ids: list[str] = []
    mismatch_details: list[dict] = []

    for record in records:
        chunk_ids: list[str] = record.get("relevant_chunk_ids", [])
        if not chunk_ids:
            logger.warning("Record with no chunk_ids: %s", record.get("question", "?"))
            missing += 1
            continue

        primary_id = chunk_ids[0]
        golden_text: str = record.get("chunk_text", "")

        try:
            result = collection.get(
                ids=[primary_id],
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.error("ChromaDB.get() failed for id=%s: %s", primary_id, exc)
            missing += 1
            missing_ids.append(primary_id)
            continue

        # ChromaDB returns empty lists when no id matches
        if not result["ids"]:
            missing += 1
            missing_ids.append(primary_id)
        else:
            found += 1
            # Check text match (first 100 chars)
            chroma_docs = result.get("documents") or []
            chroma_text: str = chroma_docs[0] if chroma_docs else ""

            if not chroma_text:
                text_missing += 1
            elif golden_text and chroma_text[:100] == golden_text[:100]:
                text_match += 1
            elif golden_text:
                text_mismatch += 1
                mismatch_details.append({
                    "chunk_id": primary_id,
                    "question": record["question"][:80],
                    "golden_first100": golden_text[:100],
                    "chroma_first100": chroma_text[:100],
                })
            else:
                # golden_text empty — skip comparison
                text_missing += 1

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GOLDEN DATASET AUDIT REPORT")
    print("=" * 70)
    print(f"  Dataset path       : {dataset_path}")
    print(f"  Total samples      : {total}")
    print(f"  ChromaDB collection: {CHROMA_COLLECTION}")
    print(f"  ChromaDB doc count : {chroma_count}")
    print()
    print("--- Chunk ID Coverage ---")
    pct_found = 100 * found / total if total else 0
    print(f"  FOUND   : {found}/{total}  ({pct_found:.1f}%)")
    print(f"  MISSING : {missing}/{total}  ({100 - pct_found:.1f}%)")
    if missing_ids:
        print(f"\n  Missing chunk IDs ({len(missing_ids)}):")
        for cid in missing_ids[:20]:
            print(f"    - {cid}")
        if len(missing_ids) > 20:
            print(f"    ... and {len(missing_ids) - 20} more")

    print()
    print("--- Text Content Verification (first 100 chars) ---")
    print(f"  MATCH    : {text_match}")
    print(f"  MISMATCH : {text_mismatch}")
    print(f"  NO TEXT  : {text_missing}  (chunk found but document empty)")
    if mismatch_details:
        print(f"\n  Mismatch details (first 5):")
        for d in mismatch_details[:5]:
            print(f"    chunk_id : {d['chunk_id']}")
            print(f"    question : {d['question']}")
            print(f"    golden   : {d['golden_first100']!r}")
            print(f"    chroma   : {d['chroma_first100']!r}")
            print()

    print()
    print("--- Question Quality ---")
    print(f"  Duplicates         : {len(duplicates)}")
    print(f"  Metadata-like Qs   : {len(metadata_questions)}")
    if metadata_questions:
        for q in metadata_questions:
            print(f"    [META] {q}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit golden dataset against ChromaDB")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=GOLDEN_DATASET_PATH,
        help=f"Path to golden_dataset.json (default: {GOLDEN_DATASET_PATH})",
    )
    args = parser.parse_args()
    audit(dataset_path=args.dataset)
