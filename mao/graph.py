"""
mao/graph.py
------------
Assembles the LangGraph StateGraph connecting all MAO agents.

Graph topology:
  START → router_node → [conditional dispatch] → agent_node → END

  The conditional edge reads state["intent"] and routes to one of:
    summarizer_node, graphrag_node, tool_node, sql_node,
    multimodal_node, code_node, critic_node

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

from mao.agents.clinical_agent import clinical_node
from mao.agents.code_agent import code_node
from mao.agents.critic_agent import critic_node
from mao.agents.graphrag_agent import graphrag_node
from mao.agents.multimodal_agent import multimodal_node
from mao.agents.router import route_to_agent, router_node
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
NODE_CODE       = "code_node"
NODE_CRITIC     = "critic_node"
NODE_CLINICAL   = "clinical_node"

_ALL_AGENT_NODES = [
    NODE_SUMMARIZER,
    NODE_GRAPHRAG,
    NODE_TOOL,
    NODE_SQL,
    NODE_MULTIMODAL,
    NODE_CODE,
    NODE_CRITIC,
    NODE_CLINICAL,
]


def build_graph() -> StateGraph:
    """
    Construct and compile the MAO LangGraph StateGraph.

    Returns a compiled graph ready for .invoke() or .astream().
    Call once at application startup and reuse the instance.
    """
    builder = StateGraph(MAOState)

    # --- Register nodes ---
    builder.add_node(NODE_ROUTER,     router_node)
    builder.add_node(NODE_SUMMARIZER, summarizer_node)
    builder.add_node(NODE_GRAPHRAG,   graphrag_node)
    builder.add_node(NODE_TOOL,       tool_node)
    builder.add_node(NODE_SQL,        sql_node)
    builder.add_node(NODE_MULTIMODAL, multimodal_node)
    builder.add_node(NODE_CODE,       code_node)
    builder.add_node(NODE_CRITIC,     critic_node)
    builder.add_node(NODE_CLINICAL,   clinical_node)

    # --- Entry point ---
    builder.add_edge(START, NODE_ROUTER)

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
            NODE_CODE:       NODE_CODE,
            NODE_CRITIC:     NODE_CRITIC,
            NODE_CLINICAL:   NODE_CLINICAL,
        },
    )

    # --- All agent nodes terminate at END ---
    for node_name in _ALL_AGENT_NODES:
        builder.add_edge(node_name, END)

    compiled = builder.compile()
    logger.info("MAO graph compiled with %d nodes", len(_ALL_AGENT_NODES) + 1)
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
