"""mao/agents/chitchat_agent.py — Fast chitchat agent, no RAG lookup."""
from __future__ import annotations

import logging

from mao.core import llm as groq_llm
from mao.core.state import MAOState

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are MAO, a friendly clinical AI assistant specialising in Alzheimer's "
    "disease and stroke research. For greetings and small talk, respond briefly "
    "and warmly, then invite the user to ask a clinical or research question."
)


def chitchat_node(state: MAOState) -> MAOState:
    """LangGraph node: respond to greetings and chitchat without RAG lookup."""
    query = state["user_query"]
    try:
        response = groq_llm.chat(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=120,
        )
    except Exception as exc:
        logger.error("Chitchat agent failed: %s", exc)
        response = "Hello! How can I help you with your clinical or research question today?"
    state["response"] = response
    state["agent_used"] = "chitchat"
    state["metadata"] = {}
    return state
