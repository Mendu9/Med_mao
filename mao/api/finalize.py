"""The single place where a graph result becomes a response the user may keep.

Two things have to happen after `graph.invoke` returns, in this order:

1. the output guardrails run — they are the last safety control, and the NLI
   and judge blocks exist only here, where the graph cannot see them;
2. the exchange is committed to long-term memory — but only if step 1 let it
   through.

Getting that order wrong is not a style question. `remember` used to be a graph
node, so it ran *before* the guardrails; text the guardrails then replaced with
the patient-safety message was already in memory, and memory is replayed as
prompt context on later turns. Expressing the order once, here, means neither
endpoint can perform half of it.
"""
from __future__ import annotations

from mao.guardrails import apply_output_guardrails
from mao.memory.interface import remember_node


async def finalize_response(state: dict, session_id: str) -> dict:
    """Apply the output guardrails, then persist what survived them."""
    state = await apply_output_guardrails(state, session_id)
    return remember_node(state)
