"""Architecture review, scope item 18 — the UI must read fields the API returns.

`HealthResponse` returns `status, groq, vector_store, postgres`.
`app/streamlit_app.py` read `health["chromadb"]` and `health["redis"]`.
Executed `GET /health` -> `['groq','postgres','status','vector_store']`, so both
UI panels permanently rendered "unknown". No test guarded it.

Asserting this by grepping the UI source would repeat the mistake the review
called out about `inspect.getsource` guards — a string check passes for the
wrong reasons. Instead the panel list is now *data* (`HEALTH_PANELS`) that the
renderer iterates, so this test binds the real structure the UI reads to the
real schema the API declares. Adding a panel for a field the API does not return
now fails here rather than in front of a clinician.
"""
from __future__ import annotations

import pytest

from app.streamlit_app import HEALTH_PANELS
from mao.api.routes.health import HealthResponse


class TestEveryPanelBindsToARealResponseField:
    @pytest.mark.parametrize("field,_label", HEALTH_PANELS)
    def test_the_field_exists_on_the_response_model(self, field: str, _label: str) -> None:
        assert field in HealthResponse.model_fields, (
            f"the health panel reads {field!r}, which /health does not return; "
            f"it returns {sorted(HealthResponse.model_fields)}"
        )

    def test_the_panels_are_not_empty(self) -> None:
        """A vacuous parametrize would make the test above pass trivially."""
        assert len(HEALTH_PANELS) >= 4

    def test_every_probed_dependency_is_surfaced(self) -> None:
        """The reverse direction: a probe nobody displays is a probe nobody reads."""
        displayed = {field for field, _ in HEALTH_PANELS}
        declared = set(HealthResponse.model_fields) - {"status"}
        assert declared == displayed, (
            f"/health reports {sorted(declared)} but the UI shows {sorted(displayed)}"
        )


class TestTheEndpointReallyReturnsThoseFields:
    """Bind against the executed payload, not only the model declaration."""

    def test_the_live_payload_carries_every_panel_field(self) -> None:
        from fastapi.testclient import TestClient

        from mao.api.routes.health import router

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as client:
            payload = client.get("/health").json()

        for field, _label in HEALTH_PANELS:
            assert field in payload, f"{field} missing from the executed /health payload"
