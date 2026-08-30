"""The single place where a graph result becomes a response the user may keep.

Three things have to happen after `graph.invoke` returns, in this order:

1. an empty body becomes an explicit failure — before the guardrails, so they
   never run their checks against nothing;
2. the output guardrails run — they are the last safety control, and the NLI
   and judge blocks exist only here, where the graph cannot see them;
3. the exchange is committed to long-term memory — but only if step 2 let it
   through.

The substituted failure in step 1 is a refusal, not clinical content, so it
correctly does not gain the clinical disclaimer — the same treatment
`apply_output_guardrails` already gives its own blocks.

Getting that order wrong is not a style question. `remember` used to be a graph
node, so it ran *before* the guardrails; text the guardrails then replaced with
the patient-safety message was already in memory, and memory is replayed as
prompt context on later turns. Expressing the order once, here, means neither
endpoint can perform half of it.

Step 1 is here for the same reason. The guard against an empty synthesis was
added to `clinical_agent` alone, so `graphrag_agent` — the route real clinical
traffic takes — still returned `""`, and an empty HTTP 200 went out. Every
downstream control waves an empty response through by design: the judge defaults
to 10, the council records `skipped: empty_response`, and the guardrails have
nothing to block. One assertion at the boundary means the next agent needs no
copy of it.
"""
from __future__ import annotations

import logging

from mao.guardrails import apply_output_guardrails
from mao.memory.interface import remember_node

logger = logging.getLogger(__name__)

#: Said plainly, because a clinician can act on it. An empty body reads as "the
#: system considered your question and had nothing to say"; what happened is
#: that generation produced nothing, which is a different fact and one they can
#: respond to by retrying.
EMPTY_RESPONSE_MESSAGE = (
    "I could not generate a response for this request. This is a system failure, "
    "not a clinical finding. Please retry, and consult a licensed clinician "
    "directly if the problem persists."
)


def _ensure_a_body(state: dict, session_id: str) -> dict:
    """A 200 has a body, whichever agent produced it.

    An agent that deferred to raw-token streaming leaves `response` empty on
    purpose and stores its prompt in `_stream_messages`; the body arrives from
    the stream, so that is not a failure.
    """
    if (state.get("response") or "").strip():
        return state
    if state.get("_stream_messages"):
        return state

    logger.warning(
        "session=%s agent=%s produced no response text — substituting an explicit "
        "failure rather than serving an empty 200",
        session_id,
        state.get("agent_used", "unknown"),
    )
    state["response"] = EMPTY_RESPONSE_MESSAGE
    # Marked blocked so the cache refuses it. Without this the substituted
    # failure is itself stored under the query key for 300s and the retry is
    # served the failure — B8's defect, one step later.
    state["output_blocked"] = True
    state["output_blocked_by"] = "empty_response"
    return state


async def finalize_response(state: dict, session_id: str) -> dict:
    """Guarantee a body, apply the output guardrails, then persist what survived."""
    state = _ensure_a_body(state, session_id)
    state = await apply_output_guardrails(state, session_id)
    return remember_node(state)
