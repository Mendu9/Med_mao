"""
mao/graph.py
------------
Assembles the LangGraph StateGraph connecting all MAO agents.

Graph topology:

    START
      |
      v
    chitchat_gate ------------------------------------.
      |  (not chitchat)                (chitchat)      |
      v                                                |
    decomposer  ->  classifier  ->  router_node        |
                                        |              |
                                        v              |
                                    risk_gate <--------'
                                        |
                        [conditional dispatch on state["intent"]]
                                        |
      .-------------------+-------------+-------------+-------------.
      v                   v             v             v             v
  summarizer_node   graphrag_node   tool_node   multimodal_node   ...
  critic_node       clinical_node   chitchat_node
      |                   |             |             |             |
      '-------------------+------ verification -------+-------------'
                                        |
                                        v
                              domain_supervisor
                                        |
                                        v
                                     council
                                   /         \\
                        senior_supervisor    blocked
                                   \\         /
                                       END

Invariants this topology exists to enforce:

  1. `risk_gate` is the ONLY edge into an agent node. Every request therefore
     carries `state["risk_level"]` — computed from the safety policy — before
     any agent runs. Agents read it to decide what they may skip; nothing may
     reach an agent without it.

  2. `verification` is the ONLY edge from an agent node into the supervision
     chain. No agent can reach `domain_supervisor` without being verified
     first, whether the request was streamed or not (P0-1).

  3. Chitchat is detected exactly once, at `chitchat_gate` (P2-14). The gate
     short-circuits the decomposer/classifier/router pipeline but still passes
     through `risk_gate`, so invariant 1 holds on that path too.

  4. There is no SQL route (P0-3): LLM-authored SQL against the operational
     database was removed rather than sandboxed.

After `senior_supervisor` or `blocked` the graph terminates. Multi-turn state is
managed by the API layer, which reinvokes the graph per request with updated
chat_history — that keeps the graph deterministic and debuggable.

Usage:
    from mao.graph import build_graph
    graph = build_graph()
    result = graph.invoke(make_initial_state("Who is Einstein?", "u1"))
    print(result["response"])
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from mao.agents.chitchat_agent import chitchat_node
from mao.agents.clinical_agent import clinical_node
from mao.agents.critic_agent import critic_node
from mao.agents.domain_classifier import classifier_node
from mao.agents.domain_supervisor import domain_supervisor_node
from mao.agents.graphrag_agent import graphrag_node
from mao.agents.llm_council import council_node
from mao.agents.multimodal_agent import multimodal_node
from mao.agents.query_decomposer import decomposer_node
from mao.agents.router import is_chitchat, route_to_agent, router_node
from mao.agents.senior_supervisor import senior_supervisor_node
from mao.agents.summarizer_agent import summarizer_node
from mao.agents.tool_agent import tool_node
from mao.core.state import INTENT_CHITCHAT, MAOState
from mao.safety.policy import get_policy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node names — kept as constants to avoid typo bugs
# ---------------------------------------------------------------------------

NODE_GATE       = "chitchat_gate"
NODE_ROUTER     = "router_node"
NODE_RISK       = "risk_gate"
NODE_VERIFY     = "verification"
NODE_SUMMARIZER = "summarizer_node"
NODE_GRAPHRAG   = "graphrag_node"
NODE_TOOL       = "tool_node"
NODE_MULTIMODAL = "multimodal_node"
NODE_CRITIC     = "critic_node"
NODE_CLINICAL   = "clinical_node"
NODE_CHITCHAT   = "chitchat_node"

_ALL_AGENT_NODES = [
    NODE_SUMMARIZER,
    NODE_GRAPHRAG,
    NODE_TOOL,
    NODE_MULTIMODAL,
    NODE_CRITIC,
    NODE_CLINICAL,
    NODE_CHITCHAT,
]

# Metadata keys that mean the request carries patient data.
_ATTACHMENT_KEYS = ("image_b64", "image_url", "report_b64", "report_path")


# ---------------------------------------------------------------------------
# Pipeline nodes owned by the graph itself
# ---------------------------------------------------------------------------

def chitchat_gate_node(state: dict) -> dict:
    """Zero-cost entry gate — the single chitchat detection site (P2-14).

    Marking the intent here lets the conditional edge skip the decomposer,
    the domain classifier, and the router LLM call entirely.
    """
    if is_chitchat(state.get("user_query", "")):
        return {**state, "intent": INTENT_CHITCHAT}
    return state


def _route_from_gate(state: dict) -> str:
    """Chitchat goes straight to the risk gate; everything else is decomposed."""
    return NODE_RISK if state.get("intent") == INTENT_CHITCHAT else "decomposer"


def risk_gate_node(state: dict) -> dict:
    """Write ``state["risk_level"]`` before any agent node can run.

    Classification is the safety policy's job, not the graph's — this node only
    supplies the two inputs (routed intent, presence of an attachment) and
    stores the verdict as a plain string so it survives JSON serialisation at
    the API boundary.
    """
    metadata = state.get("metadata") or {}
    has_attachment = any(metadata.get(key) for key in _ATTACHMENT_KEYS)
    risk = get_policy().risk_for(
        state.get("intent", ""), has_attachment=has_attachment
    )
    logger.debug(
        "Risk gate: intent=%s attachment=%s → risk=%s",
        state.get("intent", ""), has_attachment, risk.value,
    )
    return {**state, "risk_level": risk.value}


def verification_node(state: dict) -> dict:
    """Delegates to the shared verification node (mao/safety/verification.py).

    Imported lazily: that module lands on a separate branch and is wired at
    Phase 1 integration. If it is absent the node is a pass-through so the
    graph still builds; integration asserts the real implementation is present.
    """
    try:
        from mao.safety.verification import verification_node as _impl
    except ImportError:
        logger.warning(
            "mao.safety.verification is not installed — verification node is a "
            "pass-through. This is expected only before Phase 1 integration."
        )
        return state
    return _impl(state)


def blocked_response_node(state: dict) -> dict:
    verdict = state.get("council_verdict", {})
    blocked_by = verdict.get("blocked_by", "council")
    return {
        **state,
        "response": (
            f"I cannot provide this response. It was flagged by the {blocked_by} review "
            "for patient safety. Please consult a licensed clinician directly."
        ),
    }


def _route_after_council(state: dict) -> str:
    verdict = state.get("council_verdict", {})
    return "senior_supervisor" if verdict.get("passed", True) else "blocked"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the MAO LangGraph StateGraph.

    Returns a compiled graph ready for .invoke() or .astream().
    Call once at application startup and reuse the instance.
    """
    builder = StateGraph(MAOState)

    # --- Agent nodes ---
    builder.add_node(NODE_SUMMARIZER, summarizer_node)
    builder.add_node(NODE_GRAPHRAG,   graphrag_node)
    builder.add_node(NODE_TOOL,       tool_node)
    builder.add_node(NODE_MULTIMODAL, multimodal_node)
    builder.add_node(NODE_CRITIC,     critic_node)
    builder.add_node(NODE_CLINICAL,   clinical_node)
    builder.add_node(NODE_CHITCHAT,   chitchat_node)

    # --- Pre-agent pipeline ---
    builder.add_node(NODE_GATE,   chitchat_gate_node)
    builder.add_node("decomposer", decomposer_node)
    builder.add_node("classifier", classifier_node)
    builder.add_node(NODE_ROUTER, router_node)
    builder.add_node(NODE_RISK,   risk_gate_node)

    # --- Post-agent supervision pipeline ---
    builder.add_node(NODE_VERIFY,         verification_node)
    builder.add_node("domain_supervisor", domain_supervisor_node)
    builder.add_node("council",           council_node)
    builder.add_node("senior_supervisor", senior_supervisor_node)
    builder.add_node("blocked",           blocked_response_node)

    # --- Entry: chitchat_gate → (risk_gate shortcut | full pipeline) ---
    builder.add_edge(START, NODE_GATE)
    builder.add_conditional_edges(
        NODE_GATE,
        _route_from_gate,
        {NODE_RISK: NODE_RISK, "decomposer": "decomposer"},
    )
    builder.add_edge("decomposer", "classifier")
    builder.add_edge("classifier", NODE_ROUTER)

    # --- The router's verdict is priced by the policy before dispatch ---
    builder.add_edge(NODE_ROUTER, NODE_RISK)
    builder.add_conditional_edges(
        NODE_RISK,
        route_to_agent,
        {
            NODE_SUMMARIZER: NODE_SUMMARIZER,
            NODE_GRAPHRAG:   NODE_GRAPHRAG,
            NODE_TOOL:       NODE_TOOL,
            NODE_MULTIMODAL: NODE_MULTIMODAL,
            NODE_CRITIC:     NODE_CRITIC,
            NODE_CLINICAL:   NODE_CLINICAL,
            NODE_CHITCHAT:   NODE_CHITCHAT,
        },
    )

    # --- Every agent is verified before it reaches supervision (P0-1) ---
    for node_name in _ALL_AGENT_NODES:
        builder.add_edge(node_name, NODE_VERIFY)
    builder.add_edge(NODE_VERIFY, "domain_supervisor")

    builder.add_edge("domain_supervisor", "council")
    builder.add_conditional_edges(
        "council",
        _route_after_council,
        {
            "senior_supervisor": "senior_supervisor",
            "blocked":           "blocked",
        },
    )
    builder.add_edge("senior_supervisor", END)
    builder.add_edge("blocked", END)

    compiled = builder.compile()
    logger.info("MAO graph compiled with %d nodes", len(compiled.nodes))
    return compiled


# ---------------------------------------------------------------------------
# Module-level singleton — imported by api/main.py
# ---------------------------------------------------------------------------

_graph = None


def get_graph():
    """Return the compiled graph singleton, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Visual debugging helper
# ---------------------------------------------------------------------------

def print_graph_structure() -> None:
    """Print the Mermaid diagram of the graph for documentation/debugging."""
    graph = build_graph()
    try:
        print(graph.get_graph().draw_mermaid())
    except Exception:  # noqa: BLE001
        # draw_mermaid not available in all LangGraph versions
        print(f"Nodes: {sorted(graph.nodes)}")
        print(
            "Edges: chitchat_gate → [decomposer → classifier → router_node] → "
            "risk_gate → [conditional] → agent → verification → "
            "domain_supervisor → council → (senior_supervisor | blocked) → END"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_graph_structure()

    # Smoke test with a stub
    from mao.core.state import make_initial_state
    graph = get_graph()
    state = make_initial_state("What is machine learning?", "user-smoke-test")
    result = graph.invoke(state)
    print("\nResponse:", result.get("response", "")[:200])
    print("Agent used:", result.get("agent_used"))
    print("Intent:", result.get("intent"))
    print("Risk:", result.get("risk_level"))
