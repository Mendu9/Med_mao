"""arch-M2 — `mao/api/main.py` was a god module and grew against the baseline.

978 -> 1042 lines during a phase whose stated purpose was to make the system
coherent. It owned chat, streaming, four ingestion endpoints, three health
probes, usage, graph topology, feedback, report export and two eval endpoints —
about twelve responsibilities in one file, with `chat_stream_endpoint` at ~190
lines carrying `# noqa: C901` so the complexity warning was silenced rather than
addressed. The project's own rules are <800 lines per file and <50 per function,
and ingestion orchestration and evaluation do not belong under `api/` at all.

The parity test is the important one: a refactor that quietly drops an endpoint
is worse than the god module.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

# The exact route table before the split. Nothing may disappear.
EXPECTED_ROUTES = {
    ("POST", "/chat"),
    ("POST", "/chat/stream"),
    ("GET", "/eval/dashboard"),
    ("GET", "/eval/retrieval"),
    ("GET", "/export/report/{session_id}"),
    ("POST", "/feedback"),
    ("GET", "/graph"),
    ("GET", "/health"),
    ("POST", "/ingest"),
    ("POST", "/ingest/alzheimers"),
    ("POST", "/ingest/knowledge-bases"),
    ("POST", "/ingest/pubmed"),
    ("GET", "/usage"),
}


def _app_routes() -> set[tuple[str, str]]:
    from mao.api.main import app

    found: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        for method in getattr(route, "methods", set()) or set():
            if method in {"HEAD", "OPTIONS"}:
                continue
            found.add((method, path))
    return found


class TestRouteParity:
    @pytest.mark.parametrize(("method", "path"), sorted(EXPECTED_ROUTES))
    def test_route_still_exists(self, method: str, path: str) -> None:
        assert (method, path) in _app_routes()

    def test_no_unexpected_route_appeared(self) -> None:
        docs = {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect", "/metrics"}
        actual = {(m, p) for m, p in _app_routes() if p not in docs}
        assert actual == EXPECTED_ROUTES


class TestMainIsNoLongerAGodModule:
    def test_main_is_under_the_project_line_limit(self) -> None:
        from mao.api import main

        lines = len(Path(inspect.getfile(main)).read_text(encoding="utf-8").splitlines())
        assert lines < 800, f"mao/api/main.py is {lines} lines"

    def test_main_shrank_against_the_phase_baseline(self) -> None:
        """It was 978 at baseline and 1042 at the first exit gate."""
        from mao.api import main

        lines = len(Path(inspect.getfile(main)).read_text(encoding="utf-8").splitlines())
        assert lines < 978

    @pytest.mark.parametrize(
        "module_name",
        [
            "mao.api.routes.ingestion",
            "mao.api.routes.health",
            "mao.api.routes.evaluation",
            "mao.api.routes.records",
        ],
    )
    def test_the_extracted_routers_exist(self, module_name: str) -> None:
        import importlib

        module = importlib.import_module(module_name)
        assert hasattr(module, "router")

    def test_ingestion_orchestration_left_the_api_module(self) -> None:
        from mao.api import main

        source = inspect.getsource(main)
        assert "_run_ingestion" not in source
        assert "_run_alzheimers_ingestion" not in source

    def test_health_probes_left_the_api_module(self) -> None:
        from mao.api import main

        source = inspect.getsource(main)
        for probe in ("_check_groq", "_check_vector_store", "_check_postgres"):
            assert probe not in source


class TestComplexityIsAddressedNotSilenced:
    def test_the_streaming_endpoint_no_longer_silences_its_warning(self) -> None:
        from mao.api import main

        assert "noqa: C901" not in inspect.getsource(main)
