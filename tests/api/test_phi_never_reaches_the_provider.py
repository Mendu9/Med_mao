"""End-to-end: PHI must not leave the process on the report-upload path.

Adversarial review C-1 proved the leak over the real HTTP surface with a
generated PDF, and no unit test could have caught it: the scrubber was called,
it just did not scrub, because `_VALUE`'s `\\s*$` terminator meant end-of-string
without `re.MULTILINE`.

    POST /chat  metadata={"report_b64": <real PDF>}   -> HTTP 200
    site=clinical.extraction  phi_reaching_groq=['Arthur Neville Kowalczyk', ...]
    site=report.summary       phi_reaching_groq=['Arthur Neville Kowalczyk', ...]

So this test asserts at the seam that actually matters — **every message the
provider is handed** — rather than on the scrubber's return value. It builds a
real PDF, posts it through the real app, and inspects what the gateway sent.

The PDF is laid out the way a discharge summary really is: header fields
separated by narrative. A contiguous header block is what made the Wave 5
fixture pass while the system leaked.
"""
from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient

from mao.providers import gateway
from mao.providers.llm.base import ProviderResponse

# Synthetic. Names, numbers and dates are invented for this fixture.
PATIENT_NAME = "Arthur Neville Kowalczyk"
MRN = "LDS/9931/C"
NEXT_OF_KIN = "Barbara Kowalczyk"
DOB = "12/03/1948"
NHS_NUMBER = "943 476 5919"
POSTCODE = "LS16 5PH"

PHI = [PATIENT_NAME, MRN, NEXT_OF_KIN, DOB, NHS_NUMBER, POSTCODE]

# Header fields interleaved with prose — what PDF extraction of a real discharge
# summary produces once the layout collapses.
REPORT_LINES = [
    "LEEDS MEMORY SERVICE - DISCHARGE SUMMARY",
    "",
    f"Patient Name: {PATIENT_NAME}",
    "The patient was admitted following a two-week history of progressive",
    "confusion, disorientation to time, and two unwitnessed falls at home.",
    "",
    f"MRN: {MRN}",
    "MMSE was 21/30, reduced from 24/30 twelve months earlier. Bloods were",
    "unremarkable. MRI showed medial temporal lobe atrophy, Scheltens grade 3.",
    "",
    f"DOB: {DOB}",
    f"NHS Number: {NHS_NUMBER}",
    "A diagnosis of probable Alzheimer's disease was made and donepezil 5 mg",
    "once daily was commenced, to be titrated to 10 mg after four weeks.",
    "",
    f"Next of Kin: {NEXT_OF_KIN}",
    f"Address: 14 Otley Road, Leeds, {POSTCODE}",
    "Follow-up has been arranged in the memory clinic in three months.",
]


def _build_pdf() -> str:
    """A real PDF, base64-encoded, as the frontends upload it."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica", 11)
    y = 780
    for line in REPORT_LINES:
        pdf.drawString(60, y, line)
        y -= 18
    pdf.save()
    return base64.b64encode(buffer.getvalue()).decode()


class _RecordingProvider:
    """Answers plausibly, and keeps every message it was handed."""

    name = "recording"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def complete(self, *, model_id, messages, temperature, max_tokens) -> ProviderResponse:
        for message in messages:
            self.sent.append(str(message.get("content", "")))
        system = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        )
        if "VERDICT" in system:
            text = "VERDICT: PASS. No concerns."
        elif "safety" in system and "groundedness" in system:
            text = '{"safety": 10, "groundedness": 9, "notes": "ok"}'
        elif "ungrounded_claims" in system:
            text = '{"ungrounded_claims": []}'
        elif "missing" in system:
            text = '{"missing": []}'
        else:
            text = "Probable Alzheimer's disease; donepezil was commenced."
        return ProviderResponse(text=text, input_tokens=10, output_tokens=10)

    def leaked(self) -> list[str]:
        blob = "\n".join(self.sent)
        return [identifier for identifier in PHI if identifier in blob]


@pytest.fixture
def recording_provider(monkeypatch: pytest.MonkeyPatch):
    provider = _RecordingProvider()
    gateway.set_provider(provider)

    from mao.agents import clinical_agent
    from mao.memory.interface import reset_memory_store, set_memory_store
    from mao.safety import verification

    class _RecordingMemory:
        """Instrument the memory sink instead of stubbing it.

        Both PHI tests installed a `_NoMemory` whose `remember` was `return
        None`, so nothing in the suite had ever asserted what the memory store
        receives — and it is a PERSISTENT sink (hosted Qdrant) that is replayed
        as prompt context on later turns. Stubbing a sink hides it; recording it
        is what lets the four-sink guarantee actually cover five.
        """

        def __init__(self) -> None:
            self.written: list[str] = []

        def recall(self, query, user_id):
            return ""

        def remember(self, query, response, user_id):
            self.written.append(f"{query}\n{response}")
            return None

    memory = _RecordingMemory()
    provider.memory = memory
    set_memory_store(memory)
    monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
    monkeypatch.setattr(verification, "check_all_claims", lambda claims, premise: [])

    yield provider

    gateway.reset_provider()
    reset_memory_store()


@pytest.fixture
def client(recording_provider):
    from mao.api.main import app

    with TestClient(app) as test_client:
        yield test_client


class TestAnUploadedReportIsDeIdentifiedBeforeItLeavesTheProcess:
    def test_the_request_succeeds(self, client, recording_provider) -> None:
        response = client.post(
            "/chat",
            json={
                "query": "Summarise this discharge summary.",
                "user_id": "phi-probe",
                "metadata": {"report_b64": _build_pdf()},
            },
        )
        assert response.status_code == 200

    def test_no_identifier_reaches_the_provider(self, client, recording_provider) -> None:
        client.post(
            "/chat",
            json={
                "query": "Summarise this discharge summary.",
                "user_id": "phi-probe",
                "metadata": {"report_b64": _build_pdf()},
            },
        )
        assert recording_provider.sent, "the provider was never called — probe is vacuous"
        assert recording_provider.leaked() == [], (
            "these identifiers were sent to the third-party provider"
        )

    def test_no_identifier_reaches_the_memory_store(
        self, client, recording_provider
    ) -> None:
        """The memory store is a PERSISTENT sink and was never asserted on.

        The query carries the identifiers, not the PDF. `remember()` receives
        `state["user_query"]` and the response — so a probe that puts the PHI
        only in an uploaded report can never fail, whatever the scrubber does.
        A reviewer proved exactly that: with the scrubber disabled at all three
        call sites and every identifier reaching the provider in the clear,
        this assertion still saw an empty list. It is the same vacuity this
        file was written to eliminate, reintroduced by the author.

        Mutation control: neutering `apply_input_guardrails` makes this fail.
        """
        response = client.post(
            "/chat",
            json={
                "query": (
                    f"Patient Name: {PATIENT_NAME} MRN: {MRN}. "
                    "Should donepezil be titrated in sinus bradycardia?"
                ),
                "user_id": "phi-probe",
            },
        )
        assert response.status_code == 200
        written = "\n".join(recording_provider.memory.written)
        assert written.strip(), "nothing was written to memory — the probe is vacuous"
        leaked = [identifier for identifier in PHI if identifier in written]
        assert leaked == [], (
            f"these identifiers were written to the memory store: {leaked}"
        )

    def test_the_clinical_content_still_reaches_the_provider(
        self, client, recording_provider
    ) -> None:
        """De-identification must not gut the report — otherwise the answer is
        based on nothing and the test above passes for the wrong reason."""
        client.post(
            "/chat",
            json={
                "query": "Summarise this discharge summary.",
                "user_id": "phi-probe",
                "metadata": {"report_b64": _build_pdf()},
            },
        )
        blob = "\n".join(recording_provider.sent)
        assert "donepezil" in blob
        assert "Scheltens" in blob or "atrophy" in blob
