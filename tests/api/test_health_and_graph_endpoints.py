"""The health check and topology endpoint must not depend on deleted modules.

Track B removed `mao/agents/sql_agent.py` (P0-3). `_check_postgres` imported
`_get_engine` from it, so the Postgres health probe would report a permanent
ImportError and `/health` would sit at `degraded` forever. The probe belongs on
the same engine the rest of the system uses (P1-10) anyway.
"""
from __future__ import annotations

import ast
import inspect


def _imported_modules(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


class TestHealthCheckIndependence:
    def test_api_does_not_import_the_deleted_sql_agent(self) -> None:
        import mao.api.main as main

        assert not any("sql_agent" in m for m in _imported_modules(main))

    def test_postgres_probe_uses_the_shared_database_layer(self) -> None:
        import mao.api.main as main

        source = inspect.getsource(main._check_postgres)
        assert "mao.db" in source


class TestGraphEndpointIsDerived:
    def test_endpoint_delegates_to_the_topology_module(self) -> None:
        import mao.api.main as main

        assert "topology" in inspect.getsource(main.graph_topology_endpoint)

    def test_no_hardcoded_node_or_intent_literals_remain(self) -> None:
        """P1-6 — the endpoint advertised code_node/sql_node long after deletion."""
        import mao.api.main as main

        source = inspect.getsource(main)
        for stale in ('"code_node"', '"sql_node"', '"code"', '"sql"'):
            assert stale not in source, f"stale topology literal {stale} still present"
