"""Shared state schema passed between all agent nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypedDict
import uuid
from mao.core.config import TOKEN_BUDGET
from mao.safety.policy import RiskLevel

if TYPE_CHECKING:  # pragma: no cover - import kept out of the runtime cycle
    from mao.trust.inputs.boundary import ProtectedInput


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

    Populated by the graph's risk gate, before any agent node runs:
      risk_level     -- RiskLevel value as a plain string ("low"/"standard"/"high")

    Conversation context (provided by API layer):
      chat_history   -- list of {"role": str, "content": str} dicts
    """

    # --- Input ---
    user_query: str
    user_id: str

    # --- Router ---
    intent: str

    # --- Safety policy (written by the graph's risk gate before any agent) ---
    risk_level: str

    # True when the router could not classify the request (provider error, or a
    # label we do not recognise). The risk gate escalates to HIGH on this rather
    # than accepting the cheapest route, so a provider outage cannot silently
    # declassify a clinical request.
    router_failed: bool

    # --- Memory (injected by mem0_handler before agent runs) ---
    memory_context: str

    # --- Conversation history (provided by API layer) ---
    #
    # De-identified, always, on any path that came from a request. Three agents
    # read this key and hand it straight to a provider — `router`, `graphrag`
    # and `critic` — and none of them scrubbed it, because the route put the
    # caller's turns here verbatim (ADV15-8). The turns are now minted by
    # `mao.trust.inputs.boundary.protect`; `initial_state_from_protected` below
    # is the only constructor a request path may use, so there is no route-level
    # spelling of this field that can carry raw text.
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

    # --- Output verification (written by verification_node) ---
    #
    # These MUST be declared. LangGraph discards any key a node returns that the
    # state schema does not declare, and it does so silently — no error, no log.
    # `judge_scores` was written by `verification_node` and dropped here, so
    # `apply_output_guardrails` always read its `safety` default of 10 and the
    # judge's BLOCK (<5) and WARN (<7) branches were unreachable in production.
    # A judge returning {"safety": 0, "notes": "would kill the patient"} shipped
    # the answer. The node was correct; the schema severed it.
    #
    # The same gap emptied `verification_trace`, which carries `prompt_ref` — so
    # the prompt-provenance field the architecture requires was blank on every
    # trace ever emitted.
    judge_scores: dict[str, Any]
    verification_trace: dict[str, Any]

    # Senior supervisor
    completeness_ok: bool
    missing_sub_queries: list[str]

    # --- Prompt provenance from the supervision chain ---
    # Written by domain_supervisor_node and senior_supervisor_node. Undeclared,
    # they were dropped exactly as judge_scores was.
    grounding_prompt_ref: str
    completeness_prompt_ref: str

    # --- Written after the graph returns, on the same dict ---
    #
    # `apply_output_guardrails` and `invoke_with_usage` run in the API layer, so
    # LangGraph never sees these writes. They are declared anyway: this TypedDict
    # is the contract for what a request carries, `mao/api/tracing.py` and
    # `mao/memory/interface.py` both read them off it, and an undeclared key
    # would be dropped the moment any of this moved inside the graph.
    output_blocked: bool
    output_blocked_by: str
    llm_usage: dict[str, Any]

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
    # A streaming-capable agent may defer its final LLM call and store messages
    # in _stream_messages ONLY when the safety policy does not require output
    # verification for this request's risk_level (P0-1). Deferring leaves
    # `response` empty, which short-circuits the whole supervision chain, so on
    # any verified route the agent generates normally and _stream_messages stays
    # empty. Transport is never an input to a safety decision.
    _want_stream: bool
    _stream_messages: list  # list[dict] — messages to send to the provider
    _stream_model: str      # resolved model id, for the trace and for debugging
    # The capability role the deferred call would have used. The endpoint streams
    # by ROLE, not by id, so this is what it needs; `_stream_model` records what
    # that role resolved to at the time the agent deferred.
    _stream_role: str


# ---------------------------------------------------------------------------
# Intent label constants — router classifies into exactly these strings
# ---------------------------------------------------------------------------

#
# There is deliberately no "sql" intent: LLM-authored SQL against the
# operational database was removed in P0-3. There is likewise no "code"
# intent — code_agent.py was deleted and the label was dead (P2-3).

INTENT_SUMMARIZE   = "summarize"
INTENT_GRAPHRAG    = "graphrag"
INTENT_TOOL        = "tool"
INTENT_MULTIMODAL  = "multimodal"
INTENT_CRITIC      = "critic"
INTENT_CLINICAL    = "clinical"
INTENT_FALLBACK    = "fallback"
INTENT_CHITCHAT    = "chitchat"

ALL_INTENTS: list[str] = [
    INTENT_SUMMARIZE,
    INTENT_GRAPHRAG,
    INTENT_TOOL,
    INTENT_MULTIMODAL,
    INTENT_CRITIC,
    INTENT_CLINICAL,
    INTENT_FALLBACK,
    INTENT_CHITCHAT,
]


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------

def risk_level_of(state: dict) -> RiskLevel:
    """Read ``state["risk_level"]`` as a :class:`RiskLevel`.

    Fail-safe by construction: an absent or unrecognised value resolves to
    STANDARD, never LOW. LOW is the only level the policy lets skip output
    verification, so it must only ever be reached by an explicit, valid write
    from the graph's risk gate.
    """
    raw = state.get("risk_level") or ""
    try:
        return RiskLevel(raw)
    except ValueError:
        return RiskLevel.STANDARD


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def initial_state_from_protected(
    protected: ProtectedInput,
    user_id: str,
) -> MAOState:
    """The only state constructor a request path may use.

    It takes the boundary's OUTPUT, not strings, so there is no way to spell
    "build a request state" that leaves a channel un-de-identified. That is the
    whole of ADV15-8 stated as a signature: the defect was not a scrubber that
    failed, it was a second channel — `chat_history` — that the route handed to
    `make_initial_state` verbatim because `make_initial_state` accepted it.

    `make_initial_state` below still takes plain text and is still the right
    thing for a CLI demo or a unit test, where there is no request, no caller
    and no patient. `tests/api/test_history_enters_the_protected_boundary.py`
    asserts that neither chat route reaches for it.
    """
    return make_initial_state(
        user_query=protected.query.text,
        user_id=user_id,
        chat_history=protected.history_as_messages(),
    )


def make_initial_state(
    user_query: str,
    user_id: str,
    chat_history: list[dict[str, str]] | None = None,
) -> MAOState:
    """
    Build a fresh MAOState with safe defaults.

    Not for the request path — see `initial_state_from_protected`.
    """
    sid = str(uuid.uuid4())
    return MAOState(
        user_query=user_query,
        user_id=user_id,
        intent="",
        risk_level="",
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
        _stream_role="",
    )
