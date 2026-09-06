r"""ADV15-7 (Phase 1 half) — the request path had no size cap.

`ChatRequest.query` was capped at 8,000 characters while `metadata` was an
unvalidated `dict[str, Any]`, so the cap sat on the one channel that did not
need it. Measured:

    pages   request body   extracted            total CPU
      10       13,224 B    32,518 c              0.57 s
     160      194,108 B   520,318 c             34.24 s

A 194 KB request pinned one of eight worker threads for 34 seconds, and doubling
the pages roughly quadrupled the cost. Fourteen requests a minute from a single
IP — inside the 60/min rate limit, which itself fails OPEN when Redis is down —
take the pool. There is no authentication anywhere in `mao/api/`.

The caps live in `mao/trust/inputs/limits.py` and are enforced where the bytes
are read. What is asserted here is the ROUTE half: that every one of those
refusals reaches the caller as a 413 that says what to send instead, and that
`/chat` and `/chat/stream` answer it identically. They have disagreed about a
refusal before — one answered 422 where the other answered a bare 500 — and the
in-graph refusals are exactly the ones at risk, because `run_graph` turns any
exception it does not recognise into an opaque 500.

Refused, never truncated: a report answered as though it were complete when half
of it was silently dropped is the same class of failure as a deleted clinical
line, and nothing downstream can tell.
"""
from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient

from mao.api.protected_input import MAX_METADATA_KEYS
from mao.providers import gateway
from mao.trust.inputs import limits
from tests.trust.recorders import RecordingProvider

# Comfortably over MAX_EXTRACTED_TEXT_CHARS (120,000) once extracted, and small
# enough to build and read in about a second.
_LINE = (
    "The patient attended the memory clinic for review of cognitive symptoms "
    "and current therapy was discussed at length with the family present. "
)
_LINES = 900


def _oversized_pdf() -> str:
    """A real PDF whose extracted text is over the cap, as a client uploads it."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica", 6)
    y = 800
    for index in range(_LINES):
        pdf.drawString(10, y, f"{index:04d} {_LINE}")
        y -= 9
        if y < 20:
            pdf.showPage()
            pdf.setFont("Helvetica", 6)
            y = 800
    pdf.save()
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(scope="module")
def oversized_pdf() -> str:
    return _oversized_pdf()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    from mao.agents import clinical_agent
    from mao.api.main import app
    from mao.memory.interface import reset_memory_store, set_memory_store
    from tests.trust.recorders import RecordingMemory

    gateway.set_provider(RecordingProvider())
    set_memory_store(RecordingMemory())
    monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")

    with TestClient(app) as test_client:
        yield test_client

    gateway.reset_provider()
    reset_memory_store()


ROUTES = ["/chat", "/chat/stream"]


def _body(**metadata) -> dict:
    return {
        "query": "Summarise this report.",
        "user_id": "size-probe",
        "metadata": metadata,
    }


class TestAnOversizedReportIsRefusedOnBothRoutes:
    """The in-graph refusal. `_extract_pdf_text` raises it per page, so an
    oversized document costs the pages read so far and not the whole
    extraction — but it raises from inside `graph.invoke`, several frames below
    the route, which is where the 500 used to come from."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_route_answers_413(self, client, oversized_pdf, route: str) -> None:
        response = client.post(route, json=_body(report_b64=oversized_pdf))
        assert response.status_code == 413, response.text

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_caller_is_told_what_to_do(
        self, client, oversized_pdf, route: str
    ) -> None:
        detail = client.post(route, json=_body(report_b64=oversized_pdf)).json()["detail"]
        assert "extracted report text" in detail
        assert str(limits.MAX_EXTRACTED_TEXT_CHARS) in detail
        assert "smaller" in detail and "split" in detail

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_refusal_carries_no_document_content(
        self, client, oversized_pdf, route: str
    ) -> None:
        """This is raised on the path that handles patient data, it is logged,
        and it reaches an HTTP body — the exact route by which a patient's name
        previously reached the application log."""
        detail = client.post(route, json=_body(report_b64=oversized_pdf)).json()["detail"]
        assert "memory clinic" not in detail

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_report_under_the_cap_is_still_processed(
        self, client, route: str
    ) -> None:
        """Otherwise the assertions above would pass with the feature removed."""
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        pdf.drawString(60, 700, "Donepezil 5 mg once daily was commenced.")
        pdf.save()
        small = base64.b64encode(buffer.getvalue()).decode()

        response = client.post(route, json=_body(report_b64=small))
        assert response.status_code == 200, response.text


class TestAnEnormousMetadataDictIsRefusedBeforeTheAgentSeesIt:
    """The route-level bound. `metadata` was an unvalidated `dict[str, Any]`,
    and it is not free even before the graph: `CacheKeyInputs.metadata_digest()`
    serialises all of it with `json.dumps` on the request thread."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_too_many_keys_is_413(self, client, route: str) -> None:
        response = client.post(
            route, json=_body(**{f"k{i}": "v" for i in range(MAX_METADATA_KEYS + 1)})
        )
        assert response.status_code == 413, response.text
        assert "request metadata" in response.json()["detail"]

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_structure_that_is_wide_rather_than_large_is_413(
        self, client, route: str
    ) -> None:
        """Node count, not only characters.

        Twelve thousand one-character elements under a single key are twelve
        thousand characters — far under the character cap and under the key cap,
        which counts top-level keys. The cost here is the shape: the route walks
        every element, and `metadata_digest` serialises every element, before
        the graph is entered. Bounding characters alone would leave that open.
        """
        response = client.post(route, json=_body(nested=["a"] * 12_000))
        assert response.status_code == 413, response.text
        assert "request metadata" in response.json()["detail"]

    @pytest.mark.parametrize("route", ROUTES)
    def test_ordinary_metadata_is_still_accepted(self, client, route: str) -> None:
        response = client.post(
            route, json=_body(patient_id="p-1", encounter_id="e-9", modality="text")
        )
        assert response.status_code == 200, response.text
