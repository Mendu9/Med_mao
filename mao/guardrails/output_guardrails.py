import logging

from mao.guardrails.db_helper import log_guardrail_event

logger = logging.getLogger(__name__)

_PATIENT_SAFETY_BLOCK_MESSAGE = (
    "I'm unable to provide this response as it may not be safe for patient care. "
    "Please consult a qualified healthcare professional."
)


async def apply_output_guardrails(state: dict, session_id: str) -> dict:
    """Modifies or replaces state response based on safety checks."""
    nli_flags = state.get("nli_flags", [])
    if nli_flags:
        unentailed_ratio = sum(1 for f in nli_flags if not f.get("entailed")) / len(nli_flags)
        if unentailed_ratio > 0.30:
            state["response"] = (
                state.get("response", "")
                + "\n\n⚠️ Some claims could not be fully verified against retrieved sources."
            )
            log_guardrail_event(session_id, "nli_confidence", triggered=True)

    verdict = state.get("council_verdict", {})
    if verdict.get("blocked_by") == "safety":
        state["response"] = _PATIENT_SAFETY_BLOCK_MESSAGE
        log_guardrail_event(session_id, "council_safety", triggered=True)
        return state

    judge = state.get("judge_scores", {})
    if judge.get("safety", 10) < 7:
        state["response"] = _PATIENT_SAFETY_BLOCK_MESSAGE
        log_guardrail_event(
            session_id, "judge_safety", triggered=True, detail=judge.get("notes")
        )
        return state

    return state
