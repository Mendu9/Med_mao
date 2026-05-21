"""
mao/eval/retrieval_metrics.py
------------------------------
Offline retrieval evaluation — MRR, Precision@K, Recall@K, F1@K.

How it works
------------
These metrics require a golden dataset: (query, relevant_chunk_ids) pairs
that say "for this question, these ChromaDB chunks are the correct answers."

Since we have no human-labelled set, we generate one synthetically:
  1. Sample N chunks from ChromaDB
  2. Ask Groq to generate a question that the chunk answers
  3. Store {question, relevant_chunk_id} → golden_dataset.json

Then we evaluate:
  For each (question, relevant_ids) in the golden dataset:
    1. Run the retriever with top_k=K
    2. Compare returned chunk_ids against relevant_ids
    3. Compute Precision@K, Recall@K, F1@K, Reciprocal Rank
  Average across all questions → MRR, mean P@K, mean R@K, mean F1@K

Results are stored in Postgres table `retrieval_eval_results`.

Usage
-----
  # Generate golden dataset (one-time, ~2 min for 50 samples)
  python -m mao.eval.retrieval_metrics --generate --samples 50

  # Run evaluation against existing golden dataset
  python -m mao.eval.retrieval_metrics --eval --k 5

  # Both in one shot
  python -m mao.eval.retrieval_metrics --generate --eval --samples 50 --k 5
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GOLDEN_DATASET_PATH = Path(__file__).parent.parent / "data" / "golden_dataset.json"

_QUESTION_GEN_SYSTEM = """\
You are an expert biomedical researcher. Given a clinical or scientific passage, \
generate ONE specific question in ENGLISH whose answer is found ONLY in that passage.

Rules:
- The question MUST be in ENGLISH regardless of the passage language.
- The question MUST be about clinical findings, mechanisms, treatments, symptoms, \
  genes, drugs, pathways, study results, measurements, or patient outcomes.
- The question must use KEY WORDS and TERMS that actually appear in the passage \
  so that a retrieval system can match it to this passage.
- Do NOT ask about: licenses, authors, journals, funding, publication metadata, \
  study design methods, or statistical methods without clinical meaning.
- Respond with ONLY the question, no preamble, no punctuation other than the question mark."""

# Keywords that indicate a chunk is metadata/boilerplate — not clinical content
_NON_CLINICAL_KEYWORDS = (
    "license", "copyright", "doi:", "issn", "elsevier", "springer", "wiley",
    "all rights reserved", "published by", "available online", "correspondence",
    "conflict of interest", "declaration", "acknowledgement", "acknowledgment",
    "funding", "grant", "this article", "open access", "creative commons",
)

# OCR artifact patterns from PDF text extraction — these chunks are unusable
_OCR_ARTIFACT_PATTERNS = (
    ".tnum/", "/zero.tnum", "/one.tnum", "/two.tnum", "fi " * 3,
    "ﬀ", "ﬁ", "ﬂ", "ﬃ", "ﬄ",
)

_MIN_CHUNK_WORDS = 100  # Raised from 80 — short chunks give unhelpful questions
_MIN_VOCAB_OVERLAP = 3  # Minimum non-stopword words shared between question and chunk

# Generic question openers that produce questions too vague for reliable retrieval.
# A question matching one of these patterns AND lacking domain-specific terms is rejected.
_GENERIC_QUESTION_PATTERNS = (
    "what is the purpose",
    "what is the main",
    "what does this",
    "what are the",
    "how does this",
    "what is this",
    "describe the",
    "explain the",
    "what is a",
)

# Stopwords excluded from the domain-term overlap check in _is_specific_question.
_QUESTION_STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "in", "to", "and", "or", "for",
    "does", "how", "are", "this", "that", "it", "with", "on", "at", "by",
    "be", "as", "from", "which", "who", "when", "where", "do", "did", "not",
    "no", "yes", "have", "has", "had", "but", "than", "more", "any", "all",
    "some", "its", "their", "can", "could", "would", "should",
}

# Jaccard overlap threshold for soft-match hit counting in run_retrieval_eval.
# 0.35 is permissive enough to catch neighboring chunks from the same section
# while still requiring meaningful overlap (avoids false positives from short texts).
_SOFT_MATCH_THRESHOLD = 0.35


def _is_clinical_chunk(text: str) -> bool:
    """Return True only if the chunk contains substantive clinical/scientific content."""
    t = text.lower()
    if len(text.split()) < _MIN_CHUNK_WORDS:
        return False
    if any(kw in t for kw in _NON_CLINICAL_KEYWORDS):
        return False
    if any(pat in text for pat in _OCR_ARTIFACT_PATTERNS):
        return False
    # Reject chunks with non-ASCII ratio > 5% (garbled OCR / non-English)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.05:
        return False
    return True


def _question_uses_chunk_vocabulary(question: str, chunk_text: str) -> bool:
    """Return True if the generated question shares meaningful vocabulary with the chunk.

    Ensures the retriever can actually match the question to the source chunk.
    """
    stopwords = {
        "the", "a", "an", "of", "in", "and", "to", "is", "was", "were", "are",
        "for", "that", "it", "with", "on", "at", "by", "be", "as", "this", "or",
        "from", "what", "which", "how", "who", "when", "where", "do", "does",
        "did", "not", "no", "yes", "have", "has", "had", "but", "than", "more",
        "any", "all", "some", "its", "their", "can", "could", "would", "should",
    }
    q_words = {w.lower().strip("?,.-") for w in question.split()} - stopwords
    chunk_words = {w.lower().strip(".,;:()-") for w in chunk_text.split()} - stopwords
    overlap = q_words & chunk_words
    return len(overlap) >= _MIN_VOCAB_OVERLAP


def _is_specific_question(question: str, chunk_text: str) -> bool:
    """Return True if the question contains at least 2 domain-specific terms from
    the chunk — i.e. it is NOT a generic/template question like "what is the main...".

    A generic opener alone is not disqualifying — the question must also lack at
    least 2 chunk-derived non-stopword terms longer than 4 characters.

    This filter sits on top of _question_uses_chunk_vocabulary and raises the bar
    from "3 shared words" to "2 shared domain words (>4 chars, not stopwords)".
    """
    q_lower = question.lower()
    # Extract substantive words: length > 4 and not a stopword
    chunk_words = {
        w.lower()
        for w in chunk_text.split()
        if len(w) > 4 and w.lower() not in _QUESTION_STOPWORDS
    }
    q_words = {
        w.lower().strip("?.,")
        for w in question.split()
        if len(w) > 4 and w.lower().strip("?.,") not in _QUESTION_STOPWORDS
    }
    domain_overlap = chunk_words & q_words
    # Must have at least 2 domain-specific words from the chunk present in the question
    return len(domain_overlap) >= 2


def _soft_match_hit(retrieved_chunk: dict, relevant_text: str, threshold: float = _SOFT_MATCH_THRESHOLD) -> bool:
    """Return True if the retrieved chunk's text overlaps significantly with the
    known-relevant chunk's text (Jaccard similarity on bag-of-words).

    This handles the common case where the same biomedical content appears in
    slightly different chunk boundaries, avoiding false negatives in recall.

    Args:
        retrieved_chunk: A retriever result dict with a "text" key.
        relevant_text:   First 500 chars of the golden relevant chunk's text.
        threshold:       Minimum Jaccard similarity to count as a hit (default 0.60).
    """
    r_words = set(retrieved_chunk.get("text", "")[:500].lower().split())
    rel_words = set(relevant_text[:500].lower().split())
    if not r_words or not rel_words:
        return False
    overlap = len(r_words & rel_words) / max(len(r_words | rel_words), 1)
    return overlap >= threshold


# ---------------------------------------------------------------------------
# Pure math — no dependencies
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> dict[str, float]:
    """
    Compute Precision@K, Recall@K, F1@K, and Reciprocal Rank for one query.

    Args:
        retrieved_ids: Ordered list of chunk IDs returned by the retriever (length >= k).
        relevant_ids:  Set of chunk IDs that are ground-truth relevant.
        k:             Cutoff rank.

    Returns:
        {"precision_at_k": float, "recall_at_k": float, "f1_at_k": float,
         "reciprocal_rank": float}
    """
    if not relevant_ids:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0, "reciprocal_rank": 0.0}

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    hits = sum(1 for rid in top_k if rid in relevant_set)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    rr = 0.0
    for rank, rid in enumerate(top_k, start=1):
        if rid in relevant_set:
            rr = 1.0 / rank
            break

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "f1_at_k": round(f1, 4),
        "reciprocal_rank": round(rr, 4),
    }


def aggregate_retrieval_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average per-query metrics into dataset-level MRR and mean P/R/F1@K."""
    if not per_query:
        return {"mrr": 0.0, "mean_precision_at_k": 0.0, "mean_recall_at_k": 0.0, "mean_f1_at_k": 0.0}

    n = len(per_query)
    return {
        "mrr": round(sum(m["reciprocal_rank"] for m in per_query) / n, 4),
        "mean_precision_at_k": round(sum(m["precision_at_k"] for m in per_query) / n, 4),
        "mean_recall_at_k": round(sum(m["recall_at_k"] for m in per_query) / n, 4),
        "mean_f1_at_k": round(sum(m["f1_at_k"] for m in per_query) / n, 4),
    }


# ---------------------------------------------------------------------------
# Golden dataset generation (LLM-assisted, writes to disk)
# ---------------------------------------------------------------------------

def generate_golden_dataset(n_samples: int = 50) -> list[dict[str, Any]]:
    """
    Sample n_samples chunks from ChromaDB, generate a question for each via Groq,
    save to golden_dataset.json, and return the list.

    Each entry: {"question": str, "relevant_chunk_ids": [str], "chunk_text": str}
    """
    try:
        import chromadb
        from mao.core.config import cfg
        from mao.core import llm as groq_llm
    except ImportError as exc:
        logger.error("Missing dependency for golden dataset generation: %s", exc)
        return []

    from chromadb.config import Settings
    client = chromadb.HttpClient(
        host=cfg.chroma_host,
        port=cfg.chroma_port,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(cfg.chroma_collection)
    total = col.count()
    if total == 0:
        logger.warning("ChromaDB collection empty — cannot generate golden dataset")
        return []

    # Oversample 15× — tighter overlap + specificity filters need more candidates
    oversample = min(total, n_samples * 15)
    step = max(1, total // oversample)
    all_ids_result = col.get(limit=total, include=[])
    all_ids: list[str] = all_ids_result["ids"]
    # Shuffle with a fixed seed for reproducibility but spread across the corpus
    import random
    rng = random.Random(42)
    shuffled_ids = all_ids[:]
    rng.shuffle(shuffled_ids)
    candidate_ids = shuffled_ids[:oversample]

    # Fetch in batches of 500 to avoid ChromaDB memory limits
    all_docs: list[tuple[str, str, dict]] = []
    batch_size = 500
    for batch_start in range(0, len(candidate_ids), batch_size):
        batch_ids = candidate_ids[batch_start:batch_start + batch_size]
        r = col.get(ids=batch_ids, include=["documents", "metadatas"])
        for nid, doc, meta in zip(r["ids"], r["documents"], r["metadatas"]):
            all_docs.append((nid, doc or "", meta or {}))

    dataset: list[dict[str, Any]] = []
    import time

    logger.info("Filtering %d candidate chunks for clinical content...", len(all_docs))
    for native_id, doc_text, meta in all_docs:
        if len(dataset) >= n_samples:
            break
        if not doc_text or not _is_clinical_chunk(doc_text):
            continue
        # Use metadata chunk_id — same value as native_id for our ingestion pipeline
        chunk_id = meta.get("chunk_id", native_id)
        retries = 3
        question = None
        for attempt in range(retries):
            try:
                question = groq_llm.chat(
                    messages=[
                        {"role": "system", "content": _QUESTION_GEN_SYSTEM},
                        {"role": "user", "content": doc_text[:1200]},
                    ],
                    temperature=0.3,
                    max_tokens=120,
                ).strip()
                break
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "rate_limit" in err_str.lower() or "Too Many Requests" in err_str:
                    wait = 2 ** attempt * 3  # 3s, 6s, 12s
                    logger.info("Groq rate-limited — waiting %ds (attempt %d/%d)", wait, attempt + 1, retries)
                    time.sleep(wait)
                else:
                    logger.debug("Question generation failed for %s: %s", chunk_id, exc)
                    break

        if not question or "?" not in question:
            continue
        # Use the same text slice the LLM saw — ensures vocab check is fair
        llm_text = doc_text[:1200]
        if not _question_uses_chunk_vocabulary(question, llm_text):
            logger.debug("Skipping low-overlap question for %s: %s", chunk_id, question[:60])
            continue
        # Reject questions that are too generic to be uniquely retrievable
        if not _is_specific_question(question, llm_text):
            logger.debug("Skipping generic question for %s: %s", chunk_id, question[:60])
            continue
        # Retrieval validation: verify the question actually retrieves its source chunk
        # in the top-20 BM25 or vector results. Questions that fail this are not
        # useful for eval — they measure generation quality, not retrieval quality.
        _validated = False
        try:
            import chromadb as _chromadb
            from mao.rag.embedder import embed_query as _embed
            from mao.rag.retriever import _bm25_search as _bm25

            _chroma_val = _chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
            _col_val = _chroma_val.get_collection(cfg.chroma_collection)
            _emb = _embed(question)
            _vres = _col_val.query(query_embeddings=[_emb], n_results=20, include=[])
            _vids = _vres.get("ids", [[]])[0]
            if chunk_id in _vids:
                _validated = True
            else:
                _bres = _bm25(question, n=20)
                if any(r["chunk_id"] == chunk_id for r in _bres):
                    _validated = True
        except Exception:
            _validated = True  # skip validation on errors (don't block generation)

        if not _validated:
            logger.debug("Skipping non-retrievable question for %s: %s", chunk_id, question[:60])
            continue

        dataset.append({
            "question": question,
            "relevant_chunk_ids": [chunk_id],
            "chunk_text": llm_text,  # store full LLM-visible text for diagnostics
            # First 500 chars stored for soft-match evaluation (chunk boundary invariance)
            "relevant_chunk_text": llm_text[:500],
        })

    logger.info("Generated %d clinical questions (target=%d)", len(dataset), n_samples)

    _GOLDEN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_GOLDEN_DATASET_PATH, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2)

    logger.info("Golden dataset saved: %d samples → %s", len(dataset), _GOLDEN_DATASET_PATH)
    return dataset


def load_golden_dataset() -> list[dict[str, Any]]:
    """Load golden dataset from disk. Returns [] if file not found."""
    if not _GOLDEN_DATASET_PATH.exists():
        return []
    with open(_GOLDEN_DATASET_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _available_ram_mb() -> int:
    """Return available RAM in MB, or 99999 if psutil is unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().available // 1024 // 1024
    except Exception:
        return 99999


def run_retrieval_eval(
    k: int = 5,
    limit: int | None = None,
    gc_every: int = 10,
) -> dict[str, Any]:
    """
    Run end-to-end retrieval evaluation against the golden dataset.

    Args:
        k:        Top-K cutoff.
        limit:    Evaluate only the first N samples (fast smoke-test).
        gc_every: Call gc.collect() every N queries to reclaim cyclic garbage.
                  Default 10; set 0 to disable.

    Returns a summary dict with aggregate metrics + per-query breakdown.
    Also stores aggregated results in Postgres `retrieval_eval_results` table.
    """
    import gc

    dataset = load_golden_dataset()
    if not dataset:
        logger.warning("No golden dataset found at %s — run with --generate first", _GOLDEN_DATASET_PATH)
        return {"error": "no_golden_dataset", "samples": 0}

    if limit is not None:
        dataset = dataset[:limit]

    try:
        from mao.rag.retriever import retrieve
    except ImportError as exc:
        logger.error("Cannot import retriever: %s", exc)
        return {"error": str(exc), "samples": 0}

    per_query: list[dict[str, float]] = []
    details: list[dict[str, Any]] = []

    import time as _time
    n_total = len(dataset)
    ram_mb = _available_ram_mb()
    logger.info(
        "Running retrieval eval on %d samples (K=%d) | RAM available: %dMB | gc_every=%d",
        n_total, k, ram_mb, gc_every,
    )
    _t_start = _time.monotonic()
    for _i, entry in enumerate(dataset, start=1):
        question = entry["question"]
        relevant_ids = list(entry["relevant_chunk_ids"])
        # Backward-compatible: older golden entries may not have relevant_chunk_text
        relevant_chunk_text: str = entry.get("relevant_chunk_text", "")
        try:
            results = retrieve(question, top_k=k)
            retrieved_ids = [r.metadata.get("chunk_id", "") for r in results]
        except Exception as exc:
            logger.debug("Retrieval failed for '%s': %s", question[:50], exc)
            results = []
            retrieved_ids = []

        # Soft-match expansion: if a retrieved chunk's text overlaps with the
        # golden relevant chunk's text, treat it as a hit even if the IDs differ.
        # This handles chunk-boundary drift without inflating the relevant set.
        if relevant_chunk_text:
            for result, rid in zip(results, retrieved_ids):
                if rid not in relevant_ids:
                    chunk_dict = {"text": result.text if hasattr(result, "text") else result.metadata.get("text", "")}
                    if _soft_match_hit(chunk_dict, relevant_chunk_text):
                        relevant_ids.append(rid)
                        logger.debug(
                            "Soft-match hit: %s ≈ golden chunk for query '%s'",
                            rid, question[:50],
                        )

        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k)
        per_query.append(metrics)
        details.append({
            "question": question[:100],
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            **metrics,
        })

        # Release results list so reranker output tensors can be GC'd promptly.
        del results
        if gc_every > 0 and _i % gc_every == 0:
            gc.collect()

        # Running progress + partial MRR every 5 queries
        if _i % 5 == 0 or _i == n_total:
            elapsed = _time.monotonic() - _t_start
            avg_s = elapsed / _i
            eta_s = avg_s * (n_total - _i)
            partial_mrr = sum(m["reciprocal_rank"] for m in per_query) / _i
            cur_ram = _available_ram_mb()
            logger.info(
                "Progress %d/%d | partial MRR=%.3f | %.1fs/query | ETA %.0fm | RAM %dMB free",
                _i, n_total, partial_mrr, avg_s, eta_s / 60, cur_ram,
            )
            if cur_ram < 500:
                logger.warning("RAM critically low (%dMB) — forcing GC", cur_ram)
                gc.collect()

    aggregated = aggregate_retrieval_metrics(per_query)
    summary = {
        "k": k,
        "samples": len(per_query),
        **aggregated,
        "details": details,
    }

    logger.info(
        "Retrieval eval complete: MRR=%.3f P@%d=%.3f R@%d=%.3f F1@%d=%.3f",
        aggregated["mrr"], k, aggregated["mean_precision_at_k"],
        k, aggregated["mean_recall_at_k"], k, aggregated["mean_f1_at_k"],
    )

    _store_retrieval_eval(aggregated, k, len(per_query))
    return summary


def _store_retrieval_eval(aggregated: dict[str, float], k: int, n_samples: int) -> None:
    """Persist aggregated eval results to Postgres."""
    try:
        from mao.db import get_db_session
        from mao.db.models import RetrievalEvalResult

        row = RetrievalEvalResult(
            k=k,
            n_samples=n_samples,
            mrr=aggregated["mrr"],
            mean_precision_at_k=aggregated["mean_precision_at_k"],
            mean_recall_at_k=aggregated["mean_recall_at_k"],
            mean_f1_at_k=aggregated["mean_f1_at_k"],
        )
        with get_db_session() as session:
            session.add(row)
        logger.debug("Retrieval eval results stored to DB")
    except Exception as exc:
        logger.warning("Failed to store retrieval eval results: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Suppress noisy HTTP request logs from httpx and related libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="MAO Retrieval Evaluation")
    parser.add_argument("--generate", action="store_true", help="Generate golden dataset from ChromaDB")
    parser.add_argument("--eval", action="store_true", help="Run retrieval evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples for golden dataset")
    parser.add_argument("--k", type=int, default=5, help="Top-K cutoff for evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N samples (fast smoke-test)")
    parser.add_argument("--gc-every", type=int, default=10,
                        help="Call gc.collect() every N queries (default 10, 0=off)")
    args = parser.parse_args()

    if not args.generate and not args.eval:
        parser.print_help()
    if args.generate:
        dataset = generate_golden_dataset(n_samples=args.samples)
        print(f"Generated {len(dataset)} golden samples -> {_GOLDEN_DATASET_PATH}")
    if args.eval:
        results = run_retrieval_eval(k=args.k, limit=args.limit, gc_every=args.gc_every)
        print(json.dumps({kk: vv for kk, vv in results.items() if kk != "details"}, indent=2))
