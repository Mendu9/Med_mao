import logging
from enum import Enum

from mao.guardrails.db_helper import log_guardrail_event

logger = logging.getLogger(__name__)


class GuardrailSeverity(str, Enum):
    BLOCK = "block"    # Hard block — replaces response with safety message
    WARN = "warn"      # Soft warning — appends ⚠️ disclaimer to response
    INFO = "info"      # Informational — logged silently


_PATIENT_SAFETY_BLOCK_MESSAGE = (
    "I'm unable to provide this response as it may not be safe for patient care. "
    "Please consult a qualified healthcare professional."
)

_NLI_WARN_RATIO = 0.30    # WARN: >30% claims unentailed → append disclaimer
_NLI_BLOCK_RATIO = 0.70   # BLOCK: >70% claims unentailed → replace response
_JUDGE_WARN_SCORE = 7     # WARN: judge safety score 5-6 → append disclaimer
_JUDGE_BLOCK_SCORE = 5    # BLOCK: judge safety score <5 (i.e. ≤4) → replace response


async def apply_output_guardrails(state: dict, session_id: str) -> dict:
    """Apply tiered output safety checks.

    BLOCK — council safety veto, judge score <4, NLI ratio >70% → replaces response
    WARN  — judge score 4-6, NLI ratio 30-70% → appends ⚠️ disclaimer
    INFO  — (reserved for future low-severity checks)
    """
    nli_flags = state.get("nli_flags", [])
    if nli_flags:
        unentailed_ratio = sum(1 for f in nli_flags if not f.get("entailed")) / len(nli_flags)

        if unentailed_ratio > _NLI_BLOCK_RATIO:
            state["response"] = _PATIENT_SAFETY_BLOCK_MESSAGE
            log_guardrail_event(
                session_id, "nli_confidence",
                triggered=True, detail=f"unentailed_ratio={unentailed_ratio:.2f}",
                severity=GuardrailSeverity.BLOCK,
            )
            return state

        elif unentailed_ratio > _NLI_WARN_RATIO:
            _nli_disclaimer = "\n\n⚠️ Some claims could not be fully verified against retrieved sources."
            _resp = state.get("response", "")
            if _nli_disclaimer.strip() not in _resp:
                state["response"] = _resp + _nli_disclaimer
            log_guardrail_event(
                session_id, "nli_confidence",
                triggered=True, detail=f"unentailed_ratio={unentailed_ratio:.2f}",
                severity=GuardrailSeverity.WARN,
            )

    # BLOCK: council safety veto
    verdict = state.get("council_verdict", {})
    if verdict.get("blocked_by") == "safety":
        state["response"] = _PATIENT_SAFETY_BLOCK_MESSAGE
        log_guardrail_event(
            session_id, "council_safety",
            triggered=True,
            severity=GuardrailSeverity.BLOCK,
        )
        return state

    judge = state.get("judge_scores", {})
    safety_score = judge.get("safety", 10)

    if safety_score < _JUDGE_BLOCK_SCORE:
        state["response"] = _PATIENT_SAFETY_BLOCK_MESSAGE
        log_guardrail_event(
            session_id, "judge_safety",
            triggered=True, detail=f"score={safety_score}; {judge.get('notes', '')}",
            severity=GuardrailSeverity.BLOCK,
        )
        return state

    elif safety_score < _JUDGE_WARN_SCORE:
        _judge_disclaimer = "\n\n⚠️ This response has moderate confidence. Please verify with a healthcare professional."
        _resp = state.get("response", "")
        if _judge_disclaimer.strip() not in _resp:
            state["response"] = _resp + _judge_disclaimer
        log_guardrail_event(
            session_id, "judge_safety",
            triggered=True, detail=f"score={safety_score} (warn threshold)",
            severity=GuardrailSeverity.WARN,
        )

    return state
