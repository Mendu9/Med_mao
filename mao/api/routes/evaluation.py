"""Evaluation read-outs: RAGAS/feedback dashboard and retrieval metrics.

Evaluation is not a serving concern and does not belong under `api/` long term;
it lives here as its own module so the boundary is at least visible.

Every SQL statement is a static literal with no user-influenced input — the
query parameters select and regenerate, they never reach a query string.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from mao.api.executor import get_executor
from mao.db import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["evaluation"])

_METRICS_SQL = (
    "SELECT faithfulness, answer_relevancy, context_precision, context_recall,"
    " latency_ms, agent_used, created_at"
    " FROM response_metrics ORDER BY created_at DESC LIMIT 100"
)

# rating: 1 = helpful, -1 = not helpful
_FEEDBACK_SQL = "SELECT rating, COUNT(*) AS cnt FROM response_feedback GROUP BY rating"

_HISTORY_SQL = (
    "SELECT k, n_samples, mrr, mean_precision_at_k, mean_recall_at_k,"
    " mean_f1_at_k, created_at"
    " FROM retrieval_eval_results ORDER BY created_at DESC LIMIT 20"
)


@router.get("/dashboard")
async def eval_dashboard() -> dict[str, Any]:
    """RAGAS metrics and feedback summary, over the shared session factory."""
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_executor(), _query_dashboard)
    except Exception as exc:  # noqa: BLE001
        logger.error("Dashboard query failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Dashboard unavailable: {exc}"
        ) from exc


def _query_dashboard() -> dict[str, Any]:
    from sqlalchemy import text

    with get_db_session() as db:
        metrics_rows = db.execute(text(_METRICS_SQL)).fetchall()
        feedback_rows = db.execute(text(_FEEDBACK_SQL)).fetchall()

    metrics = [
        {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
            "latency_ms": r.latency_ms,
            "agent_used": r.agent_used,
        }
        for r in metrics_rows
    ]
    # Mapped to True/False keys so the Streamlit tab can display 👍/👎.
    feedback = {
        ("True" if r.rating == 1 else "False"): int(r.cnt) for r in feedback_rows
    }
    return {"metrics": metrics, "feedback": feedback}


@router.get("/retrieval")
async def eval_retrieval(
    k: int = 5, regenerate: bool = False, samples: int = 50
) -> dict[str, Any]:
    """Retrieval evaluation metrics (MRR, P@K, R@K, F1@K).

    k          — Top-K cutoff
    regenerate — regenerate the golden dataset first
    samples    — golden sample count (only used when regenerate=true)
    """
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            get_executor(), lambda: _run_retrieval_eval(k, regenerate, samples)
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval eval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_retrieval_eval(k: int, regenerate: bool, samples: int) -> dict[str, Any]:
    from mao.eval.retrieval_metrics import (
        generate_golden_dataset,
        load_golden_dataset,
        run_retrieval_eval,
    )

    if regenerate:
        generate_golden_dataset(n_samples=samples)

    dataset = load_golden_dataset()
    if not dataset:
        return {
            "status": "no_golden_dataset",
            "message": (
                "No golden dataset found. Call with ?regenerate=true&samples=50 "
                "to generate one (~2 min, costs ~50 provider calls)."
            ),
            "current": None,
            "history": [],
        }

    current = run_retrieval_eval(k=k)
    return {
        "status": "ok",
        "golden_dataset_size": len(dataset),
        "current": {kk: vv for kk, vv in current.items() if kk != "details"},
        "details": current.get("details", [])[:20],
        "history": _eval_history(),
    }


def _eval_history() -> list[dict[str, Any]]:
    """Past eval runs. A missing table degrades to an empty list, not a 500."""
    try:
        from sqlalchemy import text

        with get_db_session() as db:
            rows = db.execute(text(_HISTORY_SQL)).fetchall()
        return [
            {
                "k": r.k,
                "n_samples": r.n_samples,
                "mrr": r.mrr,
                "mean_precision_at_k": r.mean_precision_at_k,
                "mean_recall_at_k": r.mean_recall_at_k,
                "mean_f1_at_k": r.mean_f1_at_k,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load retrieval eval history: %s", exc)
        return []
