"""Streaming eligibility — the API's own half of the P0-1 guarantee.

Two independent gates now stand between a request and unverified output:

  1. The agent refuses to defer its LLM call on a route the policy requires to
     be verified, so `_stream_messages` is never populated there.
  2. This gate, at the endpoint, refuses to stream raw provider tokens unless
     the policy exempts the request anyway.

Either alone would close the audited bypass. Both together mean re-opening it
takes two independent mistakes rather than one.
"""
from __future__ import annotations

import logging

from mao.safety.policy import get_policy, resolve_risk

logger = logging.getLogger(__name__)


def may_stream_raw_tokens(state: object) -> bool:
    """Whether raw provider tokens may be streamed straight to the client.

    True only when the graph prepared streaming messages AND the safety policy
    does not require this request to be verified. Everything else ships the
    text that came out of the verification chain and the output guardrails.
    """
    if not isinstance(state, dict):
        return False

    prepared = state.get("_stream_messages") or []
    if not prepared:
        return False

    risk = resolve_risk(state)
    if get_policy().requires_verification(risk):
        # An agent prepared a raw stream for a route that must be verified.
        # That is an agent bug, not a routing preference — say so loudly and
        # fall back to the verified text rather than honouring it.
        logger.error(
            "Refusing raw token stream: risk=%s requires verification but an agent "
            "prepared _stream_messages. Falling back to verified output.",
            risk.value,
        )
        return False

    return True
