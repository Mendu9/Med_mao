"""Graph topology contract tests.

Covers:
  P0-3  the public SQL route is gone (node, module, and edges)
  P0-1  every agent node reaches ``verification`` before supervision
  risk  ``risk_level`` is written before any agent node runs
"""
from __future__ import annotations

import importlib.util

import pytest

from mao.graph import (
    NODE_RISK,
    NODE_VERIFY,
    _ALL_AGENT_NODES,
    build_graph,
    risk_gate_node,
)


@pytest.fixture(scope="module")
def topology() -> dict:
    compiled = build_graph()
    drawn = compiled.get_graph()
    return {
        "nodes": set(compiled.nodes),
        "edges": {(e.source, e.target) for e in drawn.edges},
    }


# ---------------------------------------------------------------------------
# P0-3 — public SQL routing removed
# ---------------------------------------------------------------------------

def test_graph_has_no_sql_node(topology: dict) -> None:
    assert "sql_node" not in topology["nodes"]


def test_no_edge_mentions_sql(topology: dict) -> None:
    assert not [e for e in topology["edges"] if "sql" in e[0] or "sql" in e[1]]


def test_sql_agent_module_no_longer_exists() -> None:
    assert importlib.util.find_spec("mao.agents.sql_agent") is None


# ---------------------------------------------------------------------------
# P0-1 / topology — verification sits between the agents and supervision
# ---------------------------------------------------------------------------

def test_verification_node_is_registered(topology: dict) -> None:
    assert NODE_VERIFY == "verification"
    assert NODE_VERIFY in topology["nodes"]


def test_every_agent_node_feeds_verification(topology: dict) -> None:
    missing = [n for n in _ALL_AGENT_NODES if (n, NODE_VERIFY) not in topology["edges"]]
    assert missing == []


def test_no_agent_node_bypasses_verification(topology: dict) -> None:
    bypassing = [
        n for n in _ALL_AGENT_NODES if (n, "domain_supervisor") in topology["edges"]
    ]
    assert bypassing == []


def test_verification_feeds_domain_supervisor(topology: dict) -> None:
    assert (NODE_VERIFY, "domain_supervisor") in topology["edges"]


def test_supervision_chain_order(topology: dict) -> None:
    assert ("domain_supervisor", "council") in topology["edges"]
    assert ("council", "senior_supervisor") in topology["edges"]
    assert ("council", "blocked") in topology["edges"]


# ---------------------------------------------------------------------------
# Risk propagation — risk_level is written before any agent node runs
# ---------------------------------------------------------------------------

def test_risk_gate_is_the_only_entry_to_agent_nodes(topology: dict) -> None:
    """Nothing may dispatch to an agent node except the risk gate."""
    offenders = {
        (src, dst)
        for (src, dst) in topology["edges"]
        if dst in _ALL_AGENT_NODES and src != NODE_RISK
    }
    assert offenders == set()


def test_router_feeds_the_risk_gate(topology: dict) -> None:
    assert ("router_node", NODE_RISK) in topology["edges"]


def test_chitchat_shortcut_still_passes_through_the_risk_gate(topology: dict) -> None:
    assert ("chitchat_gate", NODE_RISK) in topology["edges"]


def test_risk_gate_writes_standard_risk_for_graphrag() -> None:
    out = risk_gate_node({"intent": "graphrag", "metadata": {}})
    assert out["risk_level"] == "standard"


def test_risk_gate_writes_low_risk_for_chitchat() -> None:
    out = risk_gate_node({"intent": "chitchat", "metadata": {}})
    assert out["risk_level"] == "low"


def test_risk_gate_writes_high_risk_for_clinical() -> None:
    out = risk_gate_node({"intent": "clinical", "metadata": {}})
    assert out["risk_level"] == "high"


@pytest.mark.parametrize(
    "key", ["image_b64", "image_url", "report_b64", "report_path"]
)
def test_attachment_forces_high_risk(key: str) -> None:
    out = risk_gate_node({"intent": "graphrag", "metadata": {key: "x"}})
    assert out["risk_level"] == "high"


def test_risk_gate_writes_a_plain_string() -> None:
    """The API serialises state to JSON, so risk_level must not be an Enum member."""
    out = risk_gate_node({"intent": "graphrag", "metadata": {}})
    assert type(out["risk_level"]) is str
