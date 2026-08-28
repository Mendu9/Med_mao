"""The LLM council — three independent judges, any one of which can veto.

Two audited defects shape this module:

P0-1  The council used to skip itself whenever `_want_stream` was set and the
      response was still empty, which meant a streamed request was reviewed by
      nobody. Transport is not a safety input. The only thing that legitimately
      skips review is having nothing to review.

P1-3  An empty retrieval used to return `passed=True` without asking any judge.
      Accuracy and hallucination genuinely need retrieved context, but the
      safety judge assesses patient harm from the response alone — so it always
      runs, and an ungrounded response remains blockable on safety grounds.
"""
from __future__ import annotations

import asyncio
import logging
import re

from mao.core.config import COUNCIL_MAX_TOKENS, COUNCIL_TIMEOUT_SECONDS
from mao.prompts import get_prompt
from mao.providers import gateway
from mao.providers.registry import ModelRole
from mao.safety.policy import get_policy, resolve_risk

logger = logging.getLogger(__name__)

_SAFETY = "safety"
_HALLUCINATION = "hallucination"

# Council member -> registered prompt name. Prompt text lives in the registry so
# a verdict can cite the exact prompt version that produced it.
_MEMBER_PROMPTS = {
    "accuracy": "council.accuracy",
    _HALLUCINATION: "council.hallucination",
    _SAFETY: "council.safety",
}

# Members whose judgement is meaningful with no retrieved context (P1-3).
_CONTEXT_FREE_MEMBERS = (_SAFETY,)


def _members_for(context: str) -> tuple[str, ...]:
    """Which judges can return a meaningful verdict for this context."""
    if str(context).strip():
        return tuple(_MEMBER_PROMPTS)
    return _CONTEXT_FREE_MEMBERS


# Every council prompt declares the same output contract: "VERDICT: PASS" or
# "VERDICT: FAIL" followed by one sentence.
_VERDICT_RE = re.compile(r"VERDICT\s*[:\-]?\s*(PASS|FAIL)", re.IGNORECASE)


def parse_member_verdict(text: object) -> bool:
    """Whether one member's reply is an affirmative PASS.

    Adjudication used to be `"FAIL" in text`, which made *absence of a word* the
    approval condition: an empty completion, a content-filter refusal, hedging
    prose and even "VERDICT: UNSAFE" all passed. That is the wrong default for
    the control that `mao/safety/verification.py` cites as its fail-closed
    backstop — under a degraded (rather than crashed) provider both layers
    passed unsafe content at once.

    A judge has approved a response only if it said so in the declared format.
    Anything else — including anything unparseable — is a failure to review, and
    a failure to review is a FAIL.
    """
    if not isinstance(text, str):
        return False
    found = [m.upper() for m in _VERDICT_RE.findall(text)]
    if not found:
        return False
    # A judge that says both has not approved anything.
    return "FAIL" not in found


async def _async_call_agent(member: str, response: str, context: str) -> tuple[str, str]:
    """Call one council member through the model gateway. Returns (member, verdict)."""
    spec = get_prompt(_MEMBER_PROMPTS[member])
    user_turn = get_prompt("council.user_turn").render(context=context, response=response)
    try:
        completion = await asyncio.wait_for(
            asyncio.to_thread(
                gateway.complete,
                role=ModelRole.SAFETY_JUDGE,
                messages=[
                    {"role": "system", "content": spec.template},
                    {"role": "user", "content": user_turn},
                ],
                temperature=0.0,
                max_tokens=COUNCIL_MAX_TOKENS,
            ),
            timeout=COUNCIL_TIMEOUT_SECONDS,
        )
        return member, completion.text.strip()
    except asyncio.TimeoutError:
        logger.error("Council member %s timed out after %.1fs", member, COUNCIL_TIMEOUT_SECONDS)
        return member, "VERDICT: FAIL. Timeout."
    except Exception as exc:  # noqa: BLE001 - a judge that cannot answer must not abstain
        logger.error("Council member %s failed: %s", member, exc)
        return member, "VERDICT: FAIL. Agent error."


def _prompt_refs(members: tuple[str, ...]) -> dict[str, str]:
    """Provenance: which prompt version each verdict came from."""
    return {m: get_prompt(_MEMBER_PROMPTS[m]).trace_ref for m in members}


async def run_council_async(response: str, context: str) -> dict:
    """Run every applicable council member in parallel."""
    members = _members_for(context)
    results = await asyncio.gather(
        *(_async_call_agent(m, response, context) for m in members),
        return_exceptions=True,
    )

    verdicts: dict[str, str] = {}
    for item in results:
        if isinstance(item, BaseException):
            logger.error("Council gather exception: %s", item)
            continue
        member, verdict = item
        verdicts[member] = verdict

    # A member that produced no verdict has not passed the response — it has
    # failed to review it. Fail closed.
    for member in members:
        verdicts.setdefault(member, "VERDICT: FAIL. Missing verdict.")

    meta = {"members": list(members), "prompt_refs": _prompt_refs(members)}

    # Each member must have affirmatively approved. Report the highest-severity
    # blocker first so `blocked_by` names the reason a clinician would care about.
    approved = {m: parse_member_verdict(verdicts.get(m)) for m in members}

    for member in (_SAFETY, _HALLUCINATION):
        if member in approved and not approved[member]:
            return {**verdicts, **meta, "passed": False, "blocked_by": member}

    blocker = next((m for m in members if not approved[m]), None)
    if blocker is not None:
        return {**verdicts, **meta, "passed": False, "blocked_by": blocker}

    # Clinical pipeline: unanimous affirmative PASS, or nothing goes out.
    return {**verdicts, **meta, "passed": True, "blocked_by": None}


def run_council(response: str, context: str) -> dict:
    """Synchronous wrapper — drives the async council from an executor thread.

    council_node is always called from a ThreadPoolExecutor thread (graph.invoke
    runs in an executor). asyncio.run() creates a fresh event loop for that
    thread, drives the async gather, and closes it.
    """
    try:
        return asyncio.run(run_council_async(response, context))
    except Exception as exc:  # noqa: BLE001
        logger.error("Council run failed: %s", exc)
        # Fail-safe: broken council infrastructure must NOT pass responses through.
        return {"passed": False, "blocked_by": "council_error", "error": str(exc)}


def council_node(state: dict) -> dict:
    answer = state.get("response", state.get("answer", "")) or ""
    chunks = state.get("retrieved_docs", []) or []

    # Nothing to review. Judged on its own merits — deliberately NOT conditioned
    # on `_want_stream`, so a streamed request gets the same treatment (P0-1).
    if not answer.strip():
        return {
            **state,
            "council_verdict": {
                "passed": True,
                "blocked_by": None,
                "skipped": "empty_response",
            },
        }

    # Risk-conditioned, like the rest of verification. The architecture asks for
    # "risk-conditioned verification instead of a full council on every query";
    # this node used to run unconditionally, so a LOW-risk greeting the policy
    # explicitly exempts still paid for a SAFETY_JUDGE call.
    #
    # `resolve_risk` is the strict shared resolver: only an exact "low" is LOW,
    # and an absent or malformed value classifies from intent instead. Skipping
    # is therefore reachable only by a deliberate LOW classification.
    risk = resolve_risk(state)
    if not get_policy().requires_verification(risk):
        logger.debug("Council skipped at %s risk", risk.value)
        return {
            **state,
            "council_verdict": {
                "passed": True,
                "blocked_by": None,
                "skipped": "low_risk",
            },
        }

    # An empty context is passed through: run_council_async narrows the council
    # to the members that can still judge, rather than skipping review (P1-3).
    context = "\n".join(str(c) for c in chunks[:4])
    verdict = run_council(response=answer, context=context)
    return {**state, "council_verdict": verdict}
