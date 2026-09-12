"""Build and emit request traces.

Phase 1 captures the fields later policy learning will need. Nothing read here
influences serving, and capture failures are swallowed — a broken trace sink
must never surface to a user.

Traces are deliberately content-free: no response text, no query, no attachment
payloads. Only what was done, how it scored, and what it cost.
"""
from __future__ import annotations

import logging
from typing import Any

from mao.learning.experience_store import ExperienceStore, get_experience_store
from mao.providers.gateway import resolve
from mao.providers.registry import ModelRole
from mao.safety.policy import get_policy
from mao.schemas.trace import RetrievalTrace, ToolCallTrace, TraceSchema

logger = logging.getLogger(__name__)

# Which capability role a given intent's synthesis ran on.
_INTENT_ROLE = {
    "clinical": ModelRole.CLINICAL_SYNTHESIS,
    "multimodal": ModelRole.VISION,
    "chitchat": ModelRole.ROUTER_FAST,
}


def _role_for(intent: str) -> ModelRole:
    return _INTENT_ROLE.get(intent, ModelRole.GENERAL_SYNTHESIS)


def _safety_flags(result: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    flags: list[str] = []

    verdict = result.get("council_verdict") or {}
    blocked_by = verdict.get("blocked_by")
    if blocked_by:
        flags.append(f"council_blocked_{blocked_by}")

    nli_flags = result.get("nli_flags") or []
    if any(not f.get("entailed", True) for f in nli_flags):
        flags.append("nli_unentailed")

    # `mri_low_confidence` was raised here from `metadata["uncertainty_flag"]`.
    # Both the flag and its one producer are gone with the MRI workflow (M-3):
    # `clinical_node`'s confidence gate was the only thing in the codebase that
    # ever set `uncertainty_flag` true, so the flag named a cause that can no
    # longer occur. `uncertainty_flag` itself survives as a state/audit field
    # for a later control to set; when one exists it needs a flag named after
    # what it actually measures, not after a retired predictor.

    # The two advisory supervisors write these and nothing read them — not the
    # guardrails, not the audit row, not the trace. A signal a model was paid to
    # produce and no one consumes is waste; surfacing it here makes grounding
    # and completeness observable without giving either the power to block.
    if result.get("ungrounded_claims"):
        flags.append("ungrounded_claims")

    if result.get("completeness_ok") is False:
        flags.append("incomplete_answer")

    return flags


def _retrieval(metadata: dict[str, Any]) -> RetrievalTrace | None:
    hits = metadata.get("chunks_retrieved")
    if not hits:
        return None
    top_scores = metadata.get("top_scores") or []
    return RetrievalTrace(
        index_version=str(metadata.get("index_version", "")),
        k=int(len(metadata.get("sources") or [])),
        hits=int(hits),
        reranker_id=str(metadata.get("reranker_id", "")),
        top_score=float(top_scores[0]) if top_scores else 0.0,
        scorer=str(metadata.get("score_scorer", "")),
    )


def _tool_calls(metadata: dict[str, Any]) -> list[ToolCallTrace]:
    """Which capabilities the request actually used.

    `TraceSchema.tool_calls` was declared and populated by nothing, so a trace
    could not say whether a tool ran at all — which makes tool-level policy
    (budgets, trust tiers, failure rates) unlearnable from the trace store, and
    the learning data plane is half of this phase.

    `tool_agent` emits a content-free record per invocation; the ReAct
    transcript with the actual tool output stays in `tool_trace` and is
    deliberately not read here, because traces carry no content.
    """
    raw = metadata.get("tool_calls") or []
    if not isinstance(raw, list):
        return []
    calls: list[ToolCallTrace] = []
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("tool_id"):
            continue
        calls.append(
            ToolCallTrace(
                tool_id=str(entry["tool_id"]),
                latency_ms=float(entry.get("latency_ms") or 0.0),
                ok=bool(entry.get("ok", False)),
                error=str(entry.get("error", "")),
            )
        )
    return calls


def _prompt_ref(result: dict[str, Any]) -> str:
    """Where the prompt reference actually is.

    `verification_node` writes it into the nested `state["verification_trace"]`,
    and this module read it from the top level — so the field the architecture
    requires for prompt provenance was empty on every trace ever emitted.
    """
    top_level = result.get("prompt_ref")
    if top_level:
        return str(top_level)
    nested = result.get("verification_trace") or {}
    if isinstance(nested, dict):
        return str(nested.get("prompt_ref") or "")
    return ""


def build_trace(*, trace_id: str, result: dict[str, Any], latency_ms: float) -> TraceSchema:
    """Assemble a TraceSchema from a completed graph result."""
    metadata = result.get("metadata") or {}
    intent = str(result.get("intent") or "unknown")
    role = _role_for(intent)
    record = resolve(role)

    judge = result.get("judge_scores") or {}
    quality = {k: float(v) for k, v in judge.items() if isinstance(v, (int, float))}

    # Totals for every model call the request made, summed by the gateway.
    llm_usage = result.get("llm_usage") or {}
    if not isinstance(llm_usage, dict):
        llm_usage = {}

    return TraceSchema(
        trace_id=trace_id,
        workflow=intent,
        risk_level=str(result.get("risk_level") or ""),
        provider=record.provider,
        model_id=record.model_id,
        model_role=role.name,
        prompt_ref=_prompt_ref(result),
        policy_version=get_policy().policy_version,
        latency_ms=float(latency_ms),
        retrieval=_retrieval(metadata),
        tool_calls=_tool_calls(metadata),
        input_tokens=int(llm_usage.get("input_tokens") or 0),
        output_tokens=int(llm_usage.get("output_tokens") or 0),
        estimated_cost_usd=float(llm_usage.get("estimated_cost_usd") or 0.0),
        quality_scores=quality,
        safety_flags=_safety_flags(result, metadata),
    )


def emit_trace(
    *,
    trace_id: str,
    result: dict[str, Any],
    latency_ms: float,
    store: ExperienceStore | None = None,
) -> None:
    """Capture a trace. Never raises."""
    try:
        trace = build_trace(trace_id=trace_id, result=result, latency_ms=latency_ms)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Trace construction failed (non-fatal): %s", exc)
        return
    (store or get_experience_store()).append(trace)
