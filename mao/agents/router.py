"""
mao/agents/router.py
--------------------
LangGraph router node — classifies user intent and dispatches to an agent.

Design:
  - Calls mistral (small, fast) with a structured classification prompt
  - Returns one of the ALL_INTENTS labels (see core/state.py)
  - Mem0 context is injected BEFORE calling the LLM (always)
  - The LangGraph conditional edge reads state["intent"] to route

Intent → Agent mapping:
  summarize   → summarizer_node
  graphrag    → graphrag_node
  tool        → tool_node
  sql         → sql_node
  multimodal  → multimodal_node
  code        → code_node
  critic      → critic_node
  clinical    → clinical_node
  fallback    → graphrag_node  (safe default)

Integration points:
  - graph.py adds this as the entry node
  - All agent nodes imported and registered there
  - memory/mem0_handler.py called here for pre-classification memory injection
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mao.core.config import cfg
from mao.core import llm as groq_llm
from mao.core.state import (
    ALL_INTENTS,
    INTENT_CLINICAL,
    INTENT_FALLBACK,
    INTENT_GRAPHRAG,
    MAOState,
)
from mao.memory.mem0_handler import build_system_prompt, save_memory, search_memories

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """\
You are an intent classification router for a multi-agent AI system.
Classify the user query into EXACTLY ONE of these intent labels:

  summarize   - User wants a summary of a document or topic
  graphrag    - User asks a factual/knowledge question (who, what, where, when, why)
  tool        - User needs live web search, a calculator, or a Wikipedia lookup
  sql         - User asks about structured/tabular data, statistics, or database queries
  multimodal  - User provides or asks about an image, audio, or non-text media
  code        - User wants code written, debugged, explained, or executed
  critic      - User wants feedback, review, or evaluation of text/code/plan
  clinical    - User provides an MRI scan, brain image, or medical report for analysis;
                asks about Alzheimer's stage prediction, patient scan results, what a
                scan shows, report summarization, or clinical decision support for
                brain imaging and neurology
  fallback    - Query does not fit any above category

Rules:
  - Respond with ONLY the label word, nothing else.
  - When unsure between graphrag and tool, prefer graphrag.
  - When unsure between graphrag and clinical, prefer clinical for any medical imaging
    or patient-context questions.
  - When unsure between graphrag and summarize, check if the user provides
    a passage to summarize (summarize) or just asks a question (graphrag).
"""

_ROUTER_USER_TEMPLATE = """\
{memory_context}

Chat history (last 3 turns):
{history}

User query: {query}

Intent label:"""


# ---------------------------------------------------------------------------
# Router node — called by LangGraph as the entry node
# ---------------------------------------------------------------------------

def router_node(state: MAOState) -> MAOState:
    """
    LangGraph node: classify intent and write state["intent"].
    Image/file presence → clinical without LLM call (deterministic).
    """
    user_query: str = state["user_query"]
    user_id: str    = state["user_id"]
    metadata: dict  = state.get("metadata", {})

    # --- Deterministic routing: image or PDF present → always clinical ---
    if metadata.get("image_b64") or metadata.get("report_path"):
        logger.info("Router: image/report detected → clinical (no LLM needed)")
        state["intent"] = INTENT_CLINICAL
        state["memory_context"] = ""
        return state

    # --- Mem0 pre-hook (always runs before LLM) ---
    memory_context = search_memories(user_query, user_id)
    state["memory_context"] = memory_context

    # --- Build classification prompt ---
    history_text = _format_history(state.get("chat_history", [])[-3:])
    user_prompt = _ROUTER_USER_TEMPLATE.format(
        memory_context=f"User context:\n{memory_context}" if memory_context else "",
        history=history_text or "(none)",
        query=user_query,
    )

    system_prompt = build_system_prompt(_ROUTER_SYSTEM, memory_context="")  # no mem inject for router system

    intent = _classify(system_prompt, user_prompt)
    logger.info("Router classified '%s' → intent=%s", user_query[:60], intent)

    state["intent"] = intent
    return state


def route_to_agent(state: MAOState) -> str:
    """
    LangGraph conditional edge function.

    Reads state["intent"] and returns the node name to dispatch to.
    This function is passed to graph.add_conditional_edges().
    """
    intent = state.get("intent", INTENT_FALLBACK)
    mapping = {
        "summarize":  "summarizer_node",
        "graphrag":   "graphrag_node",
        "tool":       "tool_node",
        "sql":        "sql_node",
        "multimodal": "multimodal_node",
        "code":       "code_node",
        "critic":     "critic_node",
        "clinical":   "clinical_node",
        "fallback":   "graphrag_node",
    }
    target = mapping.get(intent, "graphrag_node")
    logger.debug("Routing intent=%s → node=%s", intent, target)
    return target


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _classify(system_prompt: str, user_prompt: str) -> str:
    """Classify intent via Groq."""
    try:
        raw = groq_llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        ).strip().lower()
        for token in raw.split():
            clean = token.strip(".,!?:;\"'")
            if clean in ALL_INTENTS:
                return clean
        logger.warning("Router returned unrecognised label '%s', using fallback", raw)
        return INTENT_FALLBACK
    except Exception as exc:
        logger.error("Router LLM call failed: %s", exc)
        return INTENT_GRAPHRAG


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
