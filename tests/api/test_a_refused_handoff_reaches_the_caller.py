r"""ADV20-1 — the product's documented refusal workflow did not work end to end.

`HandoffRefused` had NO handler anywhere in `mao/`. It is raised inside
`compile_handoff`, several frames below the route, so it fell to `run_graph`'s
generic `except Exception` (mao/api/invocation.py) and reached the caller as an
opaque HTTP 500 carrying none of the seven structured fields it names.

The refusal itself is correct and is the M-1 obligation being met: "clarify /
request structured input or refuse". What was broken is the second half — the
caller was never told that a refusal had happened, why, or what to send instead,
so the refusal -> structured-resupply loop the product documents was unreachable
at the transport.

This is exactly the `AmbiguousDocument` defect, in a second exception. That one
was also swallowed into a 500 until a route clause was added, and the lesson
recorded then was that an in-graph refusal needs BOTH an `except ... : raise` in
`run_graph` and a route clause, because either one alone still answers 500.
`/chat` and `/chat/stream` are asserted together for the same reason: they have
drifted over a refusal before, one answering 422 where the other answered a bare
500.

Section P amplified this from latent to routine: it took the refusal rate on the
realistic-letter column from 0/200 to 200/200, so the unreachable path became
the common path.

`HandoffRefused`'s message is counts, a percentage and field names BY
CONSTRUCTION — it holds no document text — which is why surfacing it in an HTTP
body is safe and why this finding was HIGH rather than CRITICAL. That property
is asserted here rather than assumed, because this is the path that handles
patient data.
"""
from __future__ import annotations

import ast
import base64
import inspect
import io
import re

import pytest
from fastapi.testclient import TestClient

from mao.providers import gateway
from mao.trust.handoff.compiler import _WANTED
from tests.trust.recorders import RecordingProvider

# A document the extractor can mostly not read. One recognisable drug line, then
# ten lines of nothing a clinical grammar can parse, so coverage falls under
# MINIMUM_COVERAGE and the thin-projection branch fires. Taken from
# tests/trust/test_handoff_coverage_is_measured.py, where it is asserted to
# raise HandoffRefused at the unit level — so this file tests the TRANSPORT of a
# refusal already proven to happen, not the refusal itself.
UNREADABLE = "Donepezil 10 mg once daily.\n" + "\n".join(
    f"qwertyuiop{n} asdfghjkl{n} zxcvbnm{n}" for n in range(10)
)

# Distinctive enough that finding it in an HTTP body proves document text
# escaped, and present in the source above.
_DOCUMENT_MARKER = "qwertyuiop"


def _pdf_of(text: str) -> str:
    """A real PDF, as a client uploads it."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica", 10)
    y = 800
    for line in text.splitlines():
        pdf.drawString(40, y, line)
        y -= 14
    pdf.save()
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(scope="module")
def unreadable_pdf() -> str:
    return _pdf_of(UNREADABLE)


@pytest.fixture(scope="module")
def ordinary_pdf() -> str:
    return _pdf_of(
        "Donepezil 10 mg once daily.\n"
        "Blood pressure 128/76.\n"
        "Atrial fibrillation, rate controlled.\n"
    )


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
        "query": "Is donepezil safe for this patient?",
        "user_id": "handoff-refusal-probe",
        "metadata": metadata,
    }


class TestARefusedHandoffReachesTheCallerAsARefusal:
    """The ADV20-1 reproduction. Before the fix every assertion in this class
    fails with 500 and an opaque `detail`."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_route_answers_422_not_500(
        self, client, unreadable_pdf, route: str
    ) -> None:
        response = client.post(route, json=_body(report_b64=unreadable_pdf))
        assert response.status_code == 422, response.text

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_caller_is_told_which_fields_would_resolve_it(
        self, client, unreadable_pdf, route: str
    ) -> None:
        """A refusal with no remedy is a wall. The seven field names are the
        whole point of the exception — they are what makes the resupply loop
        actionable — and they are precisely what the 500 dropped."""
        detail = client.post(
            route, json=_body(report_b64=unreadable_pdf)
        ).json()["detail"]
        for field in _WANTED:
            assert field in detail, f"{field!r} missing from the refusal"

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_caller_is_told_a_refusal_happened_and_why(
        self, client, unreadable_pdf, route: str
    ) -> None:
        """The WHY is the coverage percentage, and it is the only content that
        distinguishes the two refusal branches. `"structured patient fields"`
        alone comes from the fixed template and fires on both, so asserting only
        that would verify THAT a refusal happened and never why."""
        response = client.post(route, json=_body(report_b64=unreadable_pdf))
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]

        assert "structured patient fields" in detail
        assert re.search(r"\b\d{1,3}%", detail), (
            f"the refusal does not say how much was carried: {detail!r}"
        )
        assert "could be carried" in detail

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_refusal_carries_no_document_content(
        self, client, unreadable_pdf, route: str
    ) -> None:
        """`HandoffRefused` holds counts, a percentage and field names. Asserted,
        not assumed: this reaches an HTTP body and an application log, which is
        the exact route by which a patient's name previously reached the log on
        the de-identification failure path.

        IT MUST FIRST ESTABLISH THAT IT IS LOOKING AT THE REFUSAL. An absence
        assertion alone passes on the BROKEN path too — the pre-fix generic 500
        body contains no document text either, so a test that only checks for
        missing markers is green whether or not the feature exists. That made
        the file's most important assertion its weakest, and it is the assertion
        `protected_input.handoff_refused` cites as evidence for the no-PHI claim.
        """
        response = client.post(route, json=_body(report_b64=unreadable_pdf))
        assert response.status_code == 422, response.text
        detail = response.json()["detail"]
        assert "structured patient fields" in detail, (
            "this is not the refusal body, so the absence check below proves "
            "nothing"
        )

        assert _DOCUMENT_MARKER not in detail
        assert "Donepezil" not in detail

    def test_both_routes_answer_identically(self, client, unreadable_pdf) -> None:
        """`/chat` and `/chat/stream` drifted over `AmbiguousDocument` once
        already — one answered 422, the other a bare 500.

        Compared up to the correlation id, which is per-request BY DESIGN: the
        refusal carries `request_id` so a caller can quote it, so the two
        answers must differ in exactly that suffix and nowhere else.
        """
        first = client.post("/chat", json=_body(report_b64=unreadable_pdf))
        second = client.post("/chat/stream", json=_body(report_b64=unreadable_pdf))

        assert first.status_code == second.status_code == 422

        marker = "Quote this reference"

        def refusal_text(response) -> str:
            detail = response.json()["detail"]
            assert marker in detail, "the refusal dropped its correlation id"
            return detail.split(marker)[0]

        assert refusal_text(first) == refusal_text(second)
        assert first.json()["detail"] != second.json()["detail"], (
            "the correlation id must be per-request, not a constant"
        )

    @pytest.mark.parametrize("route", ROUTES)
    def test_an_ordinary_report_is_still_answered(
        self, client, ordinary_pdf, route: str
    ) -> None:
        """Non-vacuity. Every assertion above would pass if the route refused
        everything, which would be a worse product than the 500."""
        response = client.post(route, json=_body(report_b64=ordinary_pdf))
        assert response.status_code == 200, response.text


class TestTheRefusalCannotCarryDocumentTextByConstruction:
    """`handoff_refused` claims the 422 carries no document content BY
    CONSTRUCTION. The tests above only sample two documents, and sampling cannot
    establish a property of a type.

    `HandoffRefused.__init__` interpolates its `reason` argument VERBATIM. So
    "by construction" is a property of the RAISE SITES, not of the constructor —
    and nothing was checking the raise sites. A future `raise HandoffRefused(f"
    could not project {line}", _WANTED)` would put source text into an HTTP body
    and an application log, and every test above would still pass.

    This closes that by parsing the raise sites instead of trusting them. It is
    the same move the rest of this wave makes elsewhere: assert the property
    structurally rather than enumerate the cases that happen to be safe today.
    """

    def test_every_raise_site_passes_a_literal_reason(self) -> None:
        from mao.trust.handoff import compiler

        tree = ast.parse(inspect.getsource(compiler))
        raises = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and getattr(node.exc.func, "id", None) == "HandoffRefused"
        ]
        assert raises, "no raise site found — this test would pass vacuously"

        offenders = []
        for node in raises:
            reason = node.exc.args[0] if node.exc.args else None
            # A plain string literal, or an implicit concatenation of them, is
            # author-written text. An f-string is allowed ONLY if every
            # interpolation is a format of a number — the coverage percentage —
            # never a name bound to document text.
            if isinstance(reason, ast.Constant) and isinstance(reason.value, str):
                continue
            if isinstance(reason, ast.JoinedStr) and all(
                isinstance(value, ast.Constant)
                or (isinstance(value, ast.FormattedValue) and value.format_spec)
                for value in reason.values
            ):
                continue
            offenders.append(ast.unparse(node)[:120])

        assert offenders == [], (
            "HandoffRefused reason must be author-written text, not interpolated "
            f"document content — its message reaches an HTTP body and a log: {offenders}"
        )

    def test_the_type_can_hold_no_document_text(self) -> None:
        """The other half: even if a reason were safe, the exception must not
        acquire a field that carries the residue. `CompletenessReport` is held to
        the same rule for the same reason."""
        from mao.trust.handoff.compiler import HandoffRefused, _WANTED

        error = HandoffRefused("A fixed reason.", _WANTED)
        assert set(vars(error)) == {"reason", "wanted"}, (
            f"HandoffRefused grew a field: {sorted(vars(error))}"
        )
        assert all(isinstance(field, str) for field in error.wanted)
