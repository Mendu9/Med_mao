"""Tiered output safety checks — the last thing that runs before the user sees text.

Thresholds come from `mao.safety.policy`, not from local copies. Four modules
each keeping their own constants is how thresholds drift apart.

P0-2  The mandatory clinical disclaimer is re-asserted here when policy requires
      it and it is missing. `clinical_agent` appends it and `domain_supervisor`
      no longer overwrites the response, so it should already be present — this
      is the belt-and-braces second line of defence, because a HIGH-risk answer
      reaching a clinician without it is a patient-safety failure regardless of
      which upstream node dropped it.
"""
import logging
from enum import Enum

from mao.guardrails.db_helper import log_guardrail_event
from mao.safety.policy import get_policy, resolve_risk
from mao.safety.verification import CLINICAL_DISCLAIMER, DISCLAIMER_MARKER

logger = logging.getLogger(__name__)


class GuardrailSeverity(str, Enum):
    BLOCK = "block"    # Hard block — replaces response with safety message
    WARN = "warn"      # Soft warning — appends ⚠️ disclaimer to response
    INFO = "info"      # Informational — logged silently


_PATIENT_SAFETY_BLOCK_MESSAGE = (
    "I'm unable to provide this response as it may not be safe for patient care. "
    "Please consult a qualified healthcare professional."
)

_NLI_WARN_TEXT = "\n\n⚠️ Some claims could not be fully verified against retrieved sources."
_JUDGE_WARN_TEXT = (
    "\n\n⚠️ This response has moderate confidence. Please verify with a healthcare professional."
)


def _append_once(response: str, addition: str, marker: str | None = None) -> str:
    """Append `addition` unless it is already present."""
    needle = marker if marker is not None else addition.strip()
    if needle and needle in response:
        return response
    return response + addition


def _ensure_clinical_disclaimer(state: dict, session_id: str) -> None:
    """Re-assert the clinical disclaimer when policy requires it (P0-2)."""
    risk = resolve_risk(state)
    if not get_policy().requires_clinical_disclaimer(risk):
        return

    response = state.get("response", "") or ""
    if DISCLAIMER_MARKER in response:
        return

    state["response"] = response + CLINICAL_DISCLAIMER
    logger.warning(
        "session=%s clinical disclaimer was missing from a %s-risk response — re-asserted",
        session_id, risk.value,
    )
    log_guardrail_event(
        session_id, "clinical_disclaimer",
        triggered=True, detail=f"reasserted for {risk.value} risk",
        severity=GuardrailSeverity.WARN,
    )


def _block(state: dict, session_id: str, check: str, detail: str = "") -> dict:
    """Replace the response and record that a block happened.

    `output_blocked` is the signal memory persistence reads. The NLI and judge
    blocks are guardrail-only controls the graph cannot see, so without an
    explicit flag the pipeline had no way to know they fired — and blocked
    clinical text was written to long-term memory and replayed on later turns.
    """
    state["response"] = _PATIENT_SAFETY_BLOCK_MESSAGE
    state["output_blocked"] = True
    state["output_blocked_by"] = check
    log_guardrail_event(
        session_id, check,
        triggered=True, detail=detail,
        severity=GuardrailSeverity.BLOCK,
    )
    return state


async def apply_output_guardrails(state: dict, session_id: str) -> dict:
    """Apply tiered output safety checks.

    BLOCK — council safety veto, judge score below block threshold, NLI ratio
            above block ratio → replaces the response
    WARN  — judge score below warn threshold, NLI ratio above warn ratio →
            appends a ⚠️ disclaimer
    INFO  — (reserved for future low-severity checks)

    A BLOCKed response is a refusal, not clinical content, so it returns
    immediately without gaining a clinical disclaimer.

    Every exit sets `output_blocked`, so a caller can tell an answer the user may
    keep from one that was withdrawn.

    The flag is monotonic within a request: an answer already withheld upstream —
    by the council through `blocked_response_node`, or by the empty-body guard in
    `finalize_response` — stays withheld. Resetting it unconditionally here, as
    this used to, meant a council veto that these checks then passed came out
    with `output_blocked=False`, and the cache and the memory writer had no way
    to tell a withdrawal from an answer.
    """
    policy = get_policy()
    if state.get("output_blocked"):
        return state
    state["output_blocked"] = False
    state.pop("output_blocked_by", None)

    nli_flags = state.get("nli_flags", []) or []
    if nli_flags:
        unentailed_ratio = sum(1 for f in nli_flags if not f.get("entailed")) / len(nli_flags)

        if unentailed_ratio > policy.nli_block_ratio:
            return _block(
                state, session_id, "nli_confidence",
                f"unentailed_ratio={unentailed_ratio:.2f}",
            )

        elif unentailed_ratio > policy.nli_warn_ratio:
            state["response"] = _append_once(state.get("response", "") or "", _NLI_WARN_TEXT)
            log_guardrail_event(
                session_id, "nli_confidence",
                triggered=True, detail=f"unentailed_ratio={unentailed_ratio:.2f}",
                severity=GuardrailSeverity.WARN,
            )

    # BLOCK: council safety veto
    verdict = state.get("council_verdict", {}) or {}
    if verdict.get("blocked_by") == "safety":
        return _block(state, session_id, "council_safety")

    judge = state.get("judge_scores", {}) or {}
    safety_score = judge.get("safety", 10)

    if safety_score < policy.judge_block_score:
        return _block(
            state, session_id, "judge_safety",
            f"score={safety_score}; {judge.get('notes', '')}",
        )

    elif safety_score < policy.judge_warn_score:
        state["response"] = _append_once(state.get("response", "") or "", _JUDGE_WARN_TEXT)
        log_guardrail_event(
            session_id, "judge_safety",
            triggered=True, detail=f"score={safety_score} (warn threshold)",
            severity=GuardrailSeverity.WARN,
        )

    _ensure_clinical_disclaimer(state, session_id)

    return state
