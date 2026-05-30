"""Shared state schema passed between all agent nodes."""

from __future__ import annotations

from typing import Any, Optional, TypedDict
import uuid
from mao.core.config import TOKEN_BUDGET


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------

class MAOState(TypedDict, total=False):
    """
    Shared state dict passed between every LangGraph node.

    Required fields (must be present in initial state):
      user_query    -- raw user message
      user_id       -- identifies the user for Mem0 scoping

    Populated by the Mem0 pre-hook before any agent runs:
      memory_context -- formatted string of retrieved user memories

    Populated by agents:
      response       -- final assistant response text
      agent_used     -- name of the agent that produced the response
      metadata       -- arbitrary agent-specific payload (tool calls, SQL, etc.)
      error          -- non-empty string if the agent encountered an error

    Populated by the router:
      intent         -- classified intent label (matches INTENT_* constants)

    Conversation context (provided by API layer):
      chat_history   -- list of {"role": str, "content": str} dicts
    """

    # --- Input ---
    user_query: str
    user_id: str

    # --- Router ---
    intent: str

    # --- Memory (injected by mem0_handler before agent runs) ---
    memory_context: str

    # --- Conversation history (provided by API layer) ---
    chat_history: list[dict[str, str]]

    # --- Output (written by agent nodes) ---
    response: str
    agent_used: str
    metadata: dict[str, Any]
    error: str

    # Query decomposition
    sub_queries: list[str]
    domain: str

    # Report card
    report_card: Optional[dict]

    # Anti-hallucination
    nli_flags: list[dict]
    council_verdict: Optional[dict]

    # Senior supervisor
    completeness_ok: bool
    missing_sub_queries: list[str]

    # Token budget
    token_budget_remaining: int

    # Uncertainty
    uncertainty_flag: bool

    # PII
    pii_scrubbed_query: str

    # Feedback
    session_id: str

    # GraphRAG retrieved context (written by graphrag_node, read by domain_supervisor)
    retrieved_docs: list
    web_results: list

    # Supervisor outputs
    sources: list
    ungrounded_claims: list[str]

    # Streaming hint — set by /chat/stream endpoint before graph invocation.
    # When True, streaming-capable agents (graphrag, clinical, summarizer) skip
    # their final LLM call and store messages in _stream_messages so the endpoint
    # can drive Groq with stream=True for true token-by-token delivery.
    _want_stream: bool
    _stream_messages: list  # list[dict] — messages to send to Groq with stream=True
    _stream_model: str      # model name to use for streaming


# ---------------------------------------------------------------------------
# Intent label constants — router classifies into exactly these strings
# ---------------------------------------------------------------------------

INTENT_SUMMARIZE   = "summarize"
INTENT_GRAPHRAG    = "graphrag"
INTENT_TOOL        = "tool"
INTENT_SQL         = "sql"
INTENT_MULTIMODAL  = "multimodal"
INTENT_CODE        = "code"
INTENT_CRITIC      = "critic"
INTENT_CLINICAL    = "clinical"
INTENT_FALLBACK    = "fallback"
INTENT_CHITCHAT    = "chitchat"

ALL_INTENTS: list[str] = [
    INTENT_SUMMARIZE,
    INTENT_GRAPHRAG,
    INTENT_TOOL,
    INTENT_SQL,
    INTENT_MULTIMODAL,
    INTENT_CODE,
    INTENT_CRITIC,
    INTENT_CLINICAL,
    INTENT_FALLBACK,
    INTENT_CHITCHAT,
]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_initial_state(
    user_query: str,
    user_id: str,
    chat_history: list[dict[str, str]] | None = None,
) -> MAOState:
    """
    Build a fresh MAOState with safe defaults.

    Called by api/main.py before invoking the LangGraph graph.
    """
    sid = str(uuid.uuid4())
    return MAOState(
        user_query=user_query,
        user_id=user_id,
        intent="",
        memory_context="",
        chat_history=chat_history or [],
        response="",
        agent_used="",
        metadata={},
        error="",
        sub_queries=[],
        domain="alzheimer",
        report_card=None,
        nli_flags=[],
        council_verdict=None,
        completeness_ok=False,
        missing_sub_queries=[],
        token_budget_remaining=TOKEN_BUDGET,
        uncertainty_flag=False,
        pii_scrubbed_query=user_query,
        session_id=sid,
        retrieved_docs=[],
        web_results=[],
        sources=[],
        ungrounded_claims=[],
        _want_stream=False,
        _stream_messages=[],
        _stream_model="",
    )
