"""Async RAGAS scorer (faithfulness, relevancy, precision, recall) running as a background task after /chat."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

def _ls_traceable(fn):
    if not os.getenv("LANGSMITH_API_KEY"):
        return fn
    try:
        from langsmith import traceable
        return traceable(name="ragas_score_response", run_type="chain")(fn)
    except Exception:  # guard against version-incompatible langsmith installs
        return fn

# Agents that produce retrieved contexts — only these are scored
_SCOREABLE_AGENTS = {"graphrag", "clinical"}

# Faithfulness threshold for hallucination flag
_FAITHFULNESS_THRESHOLD = 0.7

# Faithfulness below this triggers retraining candidate storage
_RETRAINING_THRESHOLD = 0.6


@_ls_traceable
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
        scores = await asyncio.get_running_loop().run_in_executor(
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
    await asyncio.get_running_loop().run_in_executor(
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

    # Lightweight proxy metrics from LLM judge (stronger external model via Groq)
    try:
        from mao.eval.llm_judge import judge_response
        judge = judge_response(question=question, response=answer)
        scores["answer_length"] = judge.get("answer_length", 0)
        scores["citation_count"] = judge.get("citation_count", 0)
        scores["judge_safety"] = judge.get("safety", 10) / 10.0  # normalize to 0-1
        # New judge metrics — normalized from 0-10 to 0-1 range
        scores["coherence"] = float(judge.get("coherence", 0)) / 10.0
        scores["fluency"] = float(judge.get("fluency", 0)) / 10.0
        scores["helpfulness"] = float(judge.get("helpfulness", 0)) / 10.0
        scores["perplexity_proxy"] = float(judge.get("perplexity_proxy", 0)) / 10.0
    except Exception:  # noqa: BLE001
        pass  # proxy metrics are non-fatal — RAGAS scores are still returned

    # Log hallucination warning
    faithfulness = scores.get("faithfulness", 1.0)
    if faithfulness < _FAITHFULNESS_THRESHOLD:
        logger.warning(
            "Potential hallucination detected: request_id=%s agent=%s faithfulness=%.3f",
            request_id,
            agent_used,
            faithfulness,
        )

    # Eval feedback loop: flag for retraining when faithfulness is very low
    if faithfulness < _RETRAINING_THRESHOLD:
        await asyncio.get_running_loop().run_in_executor(
            None,
            _store_retraining_candidate,
            request_id,
            user_id,
            agent_used,
            question,
            answer,
            contexts,
            faithfulness,
            scores.get("answer_relevancy", 0.0),
            "low_faithfulness",
        )

    return scores


def _store_retraining_candidate(
    request_id: str,
    user_id: str,
    agent_used: str,
    question: str,
    answer: str,
    contexts: list[str],
    faithfulness: float,
    answer_relevancy: float,
    trigger_reason: str,
) -> None:
    """Store a poor-quality response as a retraining candidate (sync, runs in executor)."""
    try:
        from mao.db import get_db_session
        from mao.db.models import RetrainingCandidate
        with get_db_session() as db:
            db.add(RetrainingCandidate(
                request_id=request_id,
                user_id=user_id,
                agent_used=agent_used,
                question=question,
                answer=answer,
                contexts=contexts,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                trigger_reason=trigger_reason,
            ))
        logger.info(
            "Retraining candidate stored: request_id=%s faithfulness=%.3f",
            request_id,
            faithfulness,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Retraining candidate store failed (non-fatal): %s", exc)


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
        try:
            from ragas.metrics.collections import (
                Faithfulness, AnswerRelevancy,
            )
            # LLMContextPrecisionWithoutReference does not require a 'reference' column
            try:
                from ragas.metrics.collections import LLMContextPrecisionWithoutReference as _CtxPrec
            except ImportError:
                _CtxPrec = None
        except ImportError:
            from ragas.metrics import (  # type: ignore[no-redef]
                Faithfulness, AnswerRelevancy,
            )
            _CtxPrec = None
    except ImportError:
        logger.warning(
            "ragas not installed — skipping quality scoring. Run: pip install ragas"
        )
        return {}

    try:
        from mao.core.config import cfg

        # ragas 0.4+ uses llm_factory with native provider clients.
        # Fall back to deprecated LangchainLLMWrapper for ragas < 0.4.
        _ragas_llm = None
        _ragas_embeddings = None
        # ragas 0.3.x: LangchainLLMWrapper + ChatGroq (pinned to ragas==0.3.2)
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

        _ragas_llm = LangchainLLMWrapper(
            ChatGroq(api_key=cfg.groq_api_key, model=cfg.groq_judge_model, temperature=0)
        )

        _ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=cfg.embed_model)
        )

        # reference= intentionally omitted — ContextPrecision with reference
        # requires ground-truth labels not available at runtime.
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        dataset = EvaluationDataset(samples=[sample])

        metrics = [
            Faithfulness(llm=_ragas_llm),
            AnswerRelevancy(llm=_ragas_llm, embeddings=_ragas_embeddings),
        ]
        if _CtxPrec is not None:
            metrics.append(_CtxPrec(llm=_ragas_llm))

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        scores: dict[str, float] = {}
        # ragas 0.3+: _scores_dict is {metric_name: [float, ...]} (list per sample)
        # str(result) gives '{metric: mean_val, ...}' which we parse as a fallback
        raw_scores: dict = {}
        try:
            raw_scores = dict(result._scores_dict)
        except AttributeError:
            pass

        def _extract_scalar(val) -> float | None:
            """Return float from a scalar, list, or numpy value."""
            try:
                if isinstance(val, (list, tuple)) and val:
                    return float(sum(float(v) for v in val) / len(val))
                return float(val)
            except (TypeError, ValueError):
                return None

        for metric_name in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "llm_context_precision_without_reference",
        ]:
            v = raw_scores.get(metric_name)
            scalar = _extract_scalar(v) if v is not None else None
            if scalar is not None:
                key = "context_precision" if "context_precision" in metric_name else metric_name
                scores[key] = round(scalar, 4)

        logger.info("RAGAS scores: %s", scores)
        return scores

    except Exception as exc:  # noqa: BLE001
        logger.warning("RAGAS evaluation failed: %s", exc)
        return {}


async def run_full_ragas_eval(samples: list[dict]) -> dict[str, float]:
    """Run a batch RAGAS + LLM-judge evaluation over multiple samples.

    Intended for the Streamlit eval tab and offline benchmarking. Each sample
    must contain ``question``, ``answer``, and ``contexts`` (list[str]).

    Args:
        samples: List of dicts with keys ``question``, ``answer``, ``contexts``.

    Returns:
        Dict of metric_name → averaged score (0.0–1.0) across all samples.
        Returns ``{}`` if no samples are provided or all evaluations fail.

    Integration points:
        - Called from ``app/streamlit_app.py`` ``_render_eval_tab()``
        - Delegates RAGAS scoring to ``_run_ragas_sync`` (thread executor)
        - Delegates judge scoring to ``mao.eval.llm_judge.judge_response``
    """
    if not samples:
        return {}

    from mao.eval.llm_judge import judge_response  # noqa: PLC0415

    # Accumulate per-metric totals for averaging
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    def _add(key: str, val: float) -> None:
        totals[key] = totals.get(key, 0.0) + val
        counts[key] = counts.get(key, 0) + 1

    loop = asyncio.get_running_loop()

    for sample in samples:
        question: str = sample.get("question", "")
        answer: str = sample.get("answer", "")
        contexts: list[str] = sample.get("contexts", [])

        # RAGAS structural metrics (faithfulness, answer_relevancy, context_precision)
        if question and answer and contexts:
            try:
                ragas_scores = await loop.run_in_executor(
                    None, _run_ragas_sync, question, answer, contexts
                )
                for k, v in ragas_scores.items():
                    _add(k, v)
            except Exception as exc:  # noqa: BLE001
                logger.warning("run_full_ragas_eval: RAGAS failed for sample: %s", exc)

        # LLM-judge metrics (coherence, fluency, helpfulness, perplexity_proxy, safety …)
        if question and answer:
            try:
                judge = judge_response(question=question, response=answer)
                _add("answer_length", float(judge.get("answer_length", 0)))
                _add("citation_count", float(judge.get("citation_count", 0)))
                _add("judge_safety", float(judge.get("safety", 10)) / 10.0)
                _add("coherence", float(judge.get("coherence", 0)) / 10.0)
                _add("fluency", float(judge.get("fluency", 0)) / 10.0)
                _add("helpfulness", float(judge.get("helpfulness", 0)) / 10.0)
                _add("perplexity_proxy", float(judge.get("perplexity_proxy", 0)) / 10.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("run_full_ragas_eval: judge failed for sample: %s", exc)

    if not counts:
        return {}

    return {k: round(totals[k] / counts[k], 4) for k in totals}


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
