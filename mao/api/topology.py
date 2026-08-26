"""Graph topology, derived from the graph that actually runs.

The `/graph` endpoint previously returned a hand-maintained literal that had
drifted badly from reality (P1-6). Deriving it from the compiled graph and the
intent constants means it can only ever describe the live system.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import START


def _entry_node(compiled: Any) -> str:
    """The node START dispatches to."""
    try:
        for edge in compiled.get_graph().edges:
            if edge.source == START:
                return str(edge.target)
    except Exception:  # noqa: BLE001 - topology is diagnostic, never fatal
        pass
    return ""


def graph_topology() -> dict[str, Any]:
    """Describe the live graph: nodes, entry point, routing, and intents."""
    from mao.core.state import ALL_INTENTS
    from mao.graph import get_graph

    compiled = get_graph()
    return {
        "nodes": sorted(compiled.nodes),
        "entry": _entry_node(compiled),
        "routing": "conditional on state.intent",
        "intents": sorted(ALL_INTENTS),
    }
