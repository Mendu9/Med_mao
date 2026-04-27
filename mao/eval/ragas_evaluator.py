"""
mao/eval/ragas_evaluator.py
-----------------------------
Async RAGAS hallucination + quality scorer.

Runs as a background task after every /chat response for graphrag and clinical agents.
Does NOT block the user-facing response — fires via asyncio.create_task().

Metrics scored:
  faithfulness        — is the answer grounded in the retrieved chunks?
  answer_relevancy    — does the answer address the question?
  context_precision   — are retrieved chunks actually relevant?
  context_recall      — did retrieval miss important information?

Scores are stored to Postgres table `response_metrics`.

Hallucination threshold: faithfulness < 0.7 is flagged as a potential hallucination.
Prometheus gauge is updated after each scoring run.

Libraries:
  ragas      (pip install ragas)
  sqlalchemy (already a MAO dependency)

If ragas or the LLM is unavailable, the function logs a warning and returns {}
without crashing the application.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Agents that produce retrieved contexts — only these are scored
_SCOREABLE_AGENTS = {"graphrag", "clinical"}

# Faithfulness threshold for hallucination flag
_FAITHFULNESS_THRESHOLD = 0.7


async def score_response(
    question: str,
    answer: str,
    contexts: list[str],
    user_id: str,
    request_id: str,
    agent_used: str,
    latency_ms: float = 0.0,
) -> dict[str, float]:
    """
    Score a RAG response with RAGAS metrics.

    Args:
        question:    The user's original question.
        answer:      The LLM-generated response.
        contexts:    List of raw chunk texts used as context (from metadata["sources"]).
        user_id:     For logging/attribution.
        request_id:  Unique request ID for tracing.
        agent_used:  Which agent handled this (only graphrag/clinical are scored).
        latency_ms:  Request latency for storage.

    Returns:
        Dict of metric name → score (0.0–1.0), or {} if scoring failed/skipped.
    """
    if agent_used not in _SCOREABLE_AGENTS:
        return {}

    if not contexts:
        logger.debug("Skipping RAGAS scoring for request %s — no contexts", request_id)
        return {}

    try:
        scores = await asyncio.get_event_loop().run_in_executor(
            None,
            _run_ragas_sync,
            question,
            answer,
            contexts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAGAS scoring failed for request %s: %s", request_id, exc)
        return {}

    if not scores:
        return {}

    # Store to Postgres
    await asyncio.get_event_loop().run_in_executor(
        None,
        _store_metrics,
        request_id,
        user_id,
        agent_used,
        scores,
        latency_ms,
    )

    # Update Prometheus gauges
    _update_prometheus(agent_used, scores)

    # Log hallucination warning
    faithfulness = scores.get("faithfulness", 1.0)
    if faithfulness < _FAITHFULNESS_THRESHOLD:
        logger.warning(
            "Potential hallucination detected: request_id=%s agent=%s faithfulness=%.3f",
            request_id,
            agent_used,
            faithfulness,
        )

    return scores


def _run_ragas_sync(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """
    Synchronous RAGAS scoring — called in a thread executor.

    Uses ragas SingleTurnSample API (ragas >= 0.2).
    Falls back gracefully if ragas is not installed or if the LLM is unavailable.
    """
    try:
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )
    except ImportError:
        logger.warning(
            "ragas not installed — skipping quality scoring. Run: pip install ragas"
        )
        return {}

    try:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=answer,  # use answer as reference when ground truth unavailable
        )
        dataset = EvaluationDataset(samples=[sample])

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall(),
            ],
        )

        scores: dict[str, float] = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = result.get(metric_name)
            if val is not None:
                try:
                    scores[metric_name] = round(float(val), 4)
                except (TypeError, ValueError):
                    pass

        logger.info("RAGAS scores: %s", scores)
        return scores

    except Exception as exc:  # noqa: BLE001
        logger.warning("RAGAS evaluation failed: %s", exc)
        return {}


def _store_metrics(
    request_id: str,
    user_id: str,
    agent_used: str,
    scores: dict[str, float],
    latency_ms: float,
) -> None:
    """Write metrics to Postgres response_metrics table via ORM session."""
    try:
        from mao.db import get_db_session
        from mao.db.models import ResponseMetrics

        row = ResponseMetrics(
            request_id=request_id,
            user_id=user_id,
            agent_used=agent_used,
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_precision=scores.get("context_precision"),
            context_recall=scores.get("context_recall"),
            latency_ms=latency_ms,
        )
        with get_db_session() as session:
            session.add(row)
        logger.debug("Response metrics stored for request_id=%s", request_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to store response metrics: %s", exc)


def _update_prometheus(agent_used: str, scores: dict[str, float]) -> None:
    """Update Prometheus gauges with latest scores."""
    try:
        from mao.monitoring.metrics import faithfulness_gauge, hallucination_rate_gauge

        faithfulness = scores.get("faithfulness", 1.0)
        faithfulness_gauge.labels(agent=agent_used).set(faithfulness)

        is_hallucination = 1.0 if faithfulness < _FAITHFULNESS_THRESHOLD else 0.0
        hallucination_rate_gauge.labels(agent=agent_used).set(is_hallucination)
    except Exception:  # noqa: BLE001
        pass  # Prometheus not available — non-fatal
