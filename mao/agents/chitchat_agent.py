"""
mao/agents/chitchat_agent.py
-----------------------------
Handles greetings and off-topic chitchat with a minimal LLM-free response.
Registered in graph.py as the chitchat_node — reached via chitchat_gate shortcut
or the router's "chitchat" intent path.
"""

from __future__ import annotations

import logging

from mao.core.state import MAOState

logger = logging.getLogger(__name__)

_CHITCHAT_REPLIES: dict[str, str] = {
    "hi": "Hi! I'm MAO, a clinical AI assistant specialising in Alzheimer's disease and stroke research. How can I help you today?",
    "hello": "Hello! I'm MAO, a clinical AI assistant. Ask me anything about Alzheimer's, stroke, or brain health.",
    "hey": "Hey! Ready to help with clinical questions or biomedical research. What's on your mind?",
    "thanks": "You're welcome! Let me know if you have more questions.",
    "thank you": "Happy to help! Feel free to ask anything else.",
    "bye": "Goodbye! Come back anytime with clinical or research questions.",
}

_DEFAULT_REPLY = (
    "I'm MAO, a clinical AI assistant focused on Alzheimer's disease and stroke research. "
    "Ask me a medical or scientific question and I'll do my best to help."
)


def chitchat_node(state: MAOState) -> MAOState:
    """
    LangGraph node: generate a canned reply for chitchat queries.
    No LLM call — fast-path only.
    """
    query = state.get("user_query", "").strip().lower().rstrip("!?.,")
    reply = _CHITCHAT_REPLIES.get(query, _DEFAULT_REPLY)

    logger.info("chitchat_node: query=%r → canned reply", query[:40])

    return {
        **state,
        "response": reply,
        "agent_used": "chitchat",
        "sources": [],
    }
