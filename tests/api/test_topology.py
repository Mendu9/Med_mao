"""/graph must describe the graph that actually runs.

P1-6: the endpoint returned a hand-maintained list that advertised `code_node`
and intent `code` (deleted long before), omitted eight real nodes, and claimed
`entry: router_node` when the real entry is the chitchat gate. The UI consumes
this. The fix is to derive it, so it cannot drift again.
"""
from __future__ import annotations

from mao.api.topology import graph_topology


class TestDerivedFromTheLiveGraph:
    def test_nodes_match_the_compiled_graph(self) -> None:
        from mao.graph import build_graph

        compiled = build_graph()
        assert set(graph_topology()["nodes"]) == set(compiled.nodes)

    def test_intents_match_the_state_module(self) -> None:
        from mao.core.state import ALL_INTENTS

        assert set(graph_topology()["intents"]) == set(ALL_INTENTS)

    def test_entry_is_reported(self) -> None:
        assert graph_topology()["entry"]

    def test_entry_is_a_real_node(self) -> None:
        topology = graph_topology()
        assert topology["entry"] in topology["nodes"]


class TestNoHandMaintainedDrift:
    def test_topology_is_not_a_hardcoded_literal(self) -> None:
        """Guards the regression: a literal list drifts, a derivation cannot."""
        import ast
        import inspect

        import mao.api.topology as module

        source = inspect.getsource(module)
        tree = ast.parse(source)
        node_literals = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.endswith("_node")
        ]
        assert not node_literals, f"hardcoded node names found: {[n.value for n in node_literals]}"

    def test_every_advertised_intent_is_routable(self) -> None:
        """An intent the router cannot dispatch is drift by another name."""
        from mao.agents.router import route_to_agent

        topology = graph_topology()
        for intent in topology["intents"]:
            target = route_to_agent({"intent": intent})
            assert target in topology["nodes"], f"intent {intent!r} routes to unknown {target!r}"
