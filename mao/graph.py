"""
mao/graph.py
------------
Assembles the LangGraph StateGraph connecting all MAO agents.

Graph topology:
  START → router_node → [conditional dispatch] → agent_node → END

  The conditional edge reads state["intent"] and routes to one of:
    summarizer_node, graphrag_node, tool_node, sql_node,
    multimodal_node, critic_node

  After any agent node the graph terminates (END).
  For multi-turn conversation, the API layer reinvokes the graph
  on each request with updated chat_history.

Why terminate at END instead of looping back through router:
  - Keeps graph deterministic and debuggable
  - Multi-turn state is managed by the API layer (cleaner separation)
  - LangGraph checkpointing can resume from any node if needed

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
from mao.agents.router import _is_chitchat as _router_is_chitchat
from mao.agents.clinical_agent import clinical_node
from mao.agents.critic_agent import critic_node
from mao.agents.domain_classifier import classifier_node
from mao.agents.domain_supervisor import domain_supervisor_node
from mao.agents.graphrag_agent import graphrag_node
from mao.agents.llm_council import council_node
from mao.agents.multimodal_agent import multimodal_node
from mao.agents.query_decomposer import decomposer_node
from mao.agents.router import route_to_agent, router_node
from mao.agents.senior_supervisor import senior_supervisor_node
from mao.agents.sql_agent import sql_node
from mao.agents.summarizer_agent import summarizer_node
from mao.agents.tool_agent import tool_node
from mao.core.state import MAOState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node names — kept as constants to avoid typo bugs
# ---------------------------------------------------------------------------

NODE_ROUTER     = "router_node"
NODE_SUMMARIZER = "summarizer_node"
NODE_GRAPHRAG   = "graphrag_node"
NODE_TOOL       = "tool_node"
NODE_SQL        = "sql_node"
NODE_MULTIMODAL = "multimodal_node"
NODE_CRITIC     = "critic_node"
NODE_CLINICAL   = "clinical_node"
NODE_CHITCHAT   = "chitchat_node"

_ALL_AGENT_NODES = [
    NODE_SUMMARIZER,
    NODE_GRAPHRAG,
    NODE_TOOL,
    NODE_SQL,
    NODE_MULTIMODAL,
    NODE_CRITIC,
    NODE_CLINICAL,
    NODE_CHITCHAT,
]


def chitchat_gate_node(state: dict) -> dict:
    """Zero-cost entry gate — marks chitchat intent before the pipeline runs."""
    if _router_is_chitchat(state.get("user_query", "")):
        return {**state, "intent": "chitchat"}
    return state


def _route_from_gate(state: dict) -> str:
    """Skip decomposer/classifier/router for chitchat; use full pipeline otherwise."""
    return "chitchat_node" if state.get("intent") == "chitchat" else "decomposer"


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


def build_graph() -> StateGraph:
    """
    Construct and compile the MAO LangGraph StateGraph.

    Returns a compiled graph ready for .invoke() or .astream().
    Call once at application startup and reuse the instance.
    """
    builder = StateGraph(MAOState)

    # --- Register existing nodes ---
    builder.add_node(NODE_ROUTER,     router_node)
    builder.add_node(NODE_SUMMARIZER, summarizer_node)
    builder.add_node(NODE_GRAPHRAG,   graphrag_node)
    builder.add_node(NODE_TOOL,       tool_node)
    builder.add_node(NODE_SQL,        sql_node)
    builder.add_node(NODE_MULTIMODAL, multimodal_node)
    builder.add_node(NODE_CRITIC,     critic_node)
    builder.add_node(NODE_CLINICAL,   clinical_node)
    builder.add_node(NODE_CHITCHAT,   chitchat_node)

    # --- Register new pipeline nodes ---
    builder.add_node("decomposer",        decomposer_node)
    builder.add_node("classifier",        classifier_node)
    builder.add_node("domain_supervisor", domain_supervisor_node)
    builder.add_node("council",           council_node)
    builder.add_node("senior_supervisor", senior_supervisor_node)
    builder.add_node("blocked",           blocked_response_node)

    # --- Entry point: chitchat_gate → (chitchat shortcut | full pipeline) ---
    builder.add_node("chitchat_gate", chitchat_gate_node)
    builder.add_edge(START, "chitchat_gate")
    builder.add_conditional_edges(
        "chitchat_gate",
        _route_from_gate,
        {"chitchat_node": NODE_CHITCHAT, "decomposer": "decomposer"},
    )
    builder.add_edge("decomposer", "classifier")
    builder.add_edge("classifier", NODE_ROUTER)

    # --- Conditional dispatch from router ---
    builder.add_conditional_edges(
        NODE_ROUTER,
        route_to_agent,
        {
            NODE_SUMMARIZER: NODE_SUMMARIZER,
            NODE_GRAPHRAG:   NODE_GRAPHRAG,
            NODE_TOOL:       NODE_TOOL,
            NODE_SQL:        NODE_SQL,
            NODE_MULTIMODAL: NODE_MULTIMODAL,
            NODE_CRITIC:     NODE_CRITIC,
            NODE_CLINICAL:   NODE_CLINICAL,
            NODE_CHITCHAT:   NODE_CHITCHAT,
        },
    )

    # --- All agent nodes route to domain_supervisor (not END) ---
    for node_name in _ALL_AGENT_NODES:
        builder.add_edge(node_name, "domain_supervisor")

    # --- Post-agent supervision pipeline ---
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
        print(f"Nodes: {[NODE_ROUTER] + _ALL_AGENT_NODES}")
        print("Edges: router_node → [conditional] → agent_node → END")


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
