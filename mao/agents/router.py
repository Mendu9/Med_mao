"""Classifies user intent and routes to the appropriate specialist agent."""

from __future__ import annotations

import logging
import re

from mao.core.state import (
    ALL_INTENTS,
    INTENT_CLINICAL,
    INTENT_FALLBACK,
    INTENT_GRAPHRAG,
    MAOState,
)
from mao.memory.mem0_handler import build_system_prompt
from mao.prompts import get_prompt
from mao.providers import gateway
from mao.trust.egress.policy import EgressPurpose
from mao.providers.registry import ModelRole
from mao.safety.policy import has_attachment

logger = logging.getLogger(__name__)

# Token budget for the one-word intent label, with room for a short preamble.
_CLASSIFY_MAX_TOKENS = 64

# ---------------------------------------------------------------------------
# Chitchat detection — the ONE deterministic detection site (P2-14).
#
# `is_chitchat` is the single predicate. It is evaluated exactly once per
# request, by the graph's entry gate (mao/graph.py::chitchat_gate_node), which
# short-circuits the decomposer/classifier/router pipeline entirely. The router
# deliberately does NOT re-run it: a second detection site is a second place for
# the rule to drift.
# ---------------------------------------------------------------------------

_CHITCHAT_PATTERN = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|good|great|sure|yes|no|got it|understood)[!?.]*$",
    re.IGNORECASE,
)


def is_chitchat(query: str) -> bool:
    """Return True only if the query exactly matches a greeting/chitchat pattern."""
    return bool(_CHITCHAT_PATTERN.match(query.strip()))


# ---------------------------------------------------------------------------
# Router node — called by LangGraph as the entry node
# ---------------------------------------------------------------------------

def router_node(state: MAOState) -> MAOState:
    """
    LangGraph node: classify intent and write state["intent"].
    Image/file presence → clinical without LLM call (deterministic).
    """
    user_query: str = state["user_query"]
    metadata: dict  = state.get("metadata", {})

    # --- Deterministic routing: any attachment → always clinical ---
    # Key list is owned by the safety policy so the router, the risk gate, and
    # the agents cannot drift apart about what counts as patient data.
    if has_attachment(metadata):
        logger.info("Router: attachment detected → clinical (no LLM needed)")
        state["intent"] = INTENT_CLINICAL
        state["memory_context"] = ""
        return state

    # Chitchat is NOT re-detected here — see the module note above (P2-14).

    # Memory was recalled once by the graph's `recall` node, which runs before
    # this one. The router reads it; it does not fetch it.
    memory_context: str = state.get("memory_context", "")

    # --- Build classification prompt from the registry ---
    #
    # The history reaches a provider two statements below this one. It carried
    # PHI verbatim until the route stopped handing `make_initial_state` the
    # caller's raw turns (ADV15-8); every turn on this key is now minted by the
    # protected input boundary. This node does not re-scrub it — a second pass
    # over already-transformed text is what destroyed a clinical line in the
    # decomposer, and `mao.trust.inputs.boundary` explains why there is exactly
    # one transformation per channel.
    history_text = _format_history(state.get("chat_history", [])[-3:])
    user_prompt = get_prompt("router.user_turn").render(
        memory_context=f"User context:\n{memory_context}" if memory_context else "",
        history=history_text or "(none)",
        query=user_query,
    )

    # No memory injection into the router system prompt — the router classifies,
    # it does not answer.
    system_prompt = build_system_prompt(
        get_prompt("router.classify").template, memory_context=""
    )

    intent, ok = _classify(system_prompt, user_prompt)
    logger.info("Router classified '%s' → intent=%s (ok=%s)", user_query[:60], intent, ok)

    state["intent"] = intent
    # Surfaced so the risk gate can escalate rather than silently accept the
    # cheapest route. A provider outage must not declassify a clinical request.
    state["router_failed"] = not ok
    return state


def route_to_agent(state: MAOState) -> str:
    """
    LangGraph conditional edge function.

    Reads state["intent"] and returns the node name to dispatch to.
    This function is passed to graph.add_conditional_edges().
    """
    intent = state.get("intent", INTENT_FALLBACK)
    # No structured-data route: LLM-authored SQL against the operational
    # database was removed in P0-3.
    mapping = {
        "summarize":  "summarizer_node",
        "graphrag":   "graphrag_node",
        "tool":       "tool_node",
        # Modality is a capability, not a top-level agent (01_ARCHITECTURE.md:
        # "avoid top-level Multimodal Agent when modality can be handled as
        # workflow capability"). `clinical_node` owns image, report and audio,
        # and every attachment is forced here anyway — the multimodal node could
        # only ever be entered with nothing attached, so its only possible reply
        # was "please provide an image".
        "multimodal": "clinical_node",
        "critic":     "critic_node",
        "clinical":   "clinical_node",
        "chitchat":   "chitchat_node",
        "fallback":   "graphrag_node",
    }
    if intent not in mapping:
        # Silently mapping the unknown to graphrag is how "CLINICAL", "clinical "
        # and None all reached the one route that carries no clinical
        # disclaimer. `_classify` validates against ALL_INTENTS before writing,
        # so reaching here means something upstream wrote a value it should not
        # have — a bug to surface, not to absorb.
        raise ValueError(f"unroutable intent {intent!r}")
    target = mapping[intent]
    logger.debug("Routing intent=%s → node=%s", intent, target)
    return target


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _classify(system_prompt: str, user_prompt: str) -> tuple[str, bool]:
    """Classify intent. Returns (intent, classification_succeeded).

    The second element distinguishes "the model answered and I understood it"
    from "I fell back". A caller that cannot tell the difference has no way to
    know a request was never really classified — which is how a provider outage
    quietly downgraded clinical requests to the cheapest route.
    """
    try:
        raw = gateway.complete(
            role=ModelRole.ROUTER_FAST,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            purpose=EgressPurpose.ROUTING,
            temperature=0.0,
            # Enough headroom for a model that emits a short preamble before the
            # label. At 10 a reasoning model spends the whole budget thinking and
            # returns an empty string, which reads as "never classified" — and
            # that escalates every request to HIGH risk.
            max_tokens=_CLASSIFY_MAX_TOKENS,
        ).text.strip().lower()
        for token in raw.split():
            clean = token.strip(".,!?:;\"'")
            if clean in ALL_INTENTS:
                return clean, True
        # The model replied but said nothing we recognise — we did not classify.
        logger.warning("Router returned unrecognised label '%s', using fallback", raw)
        return INTENT_FALLBACK, False
    except Exception as exc:
        logger.error("Router LLM call failed: %s", exc)
        return INTENT_GRAPHRAG, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history(history: list[dict[str, str]]) -> str:
    lines = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state("Summarise this article for me: [long text]", "user-test")
    state = router_node(state)
    print("Intent:", state["intent"])
