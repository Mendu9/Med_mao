r"""ADV15-15 — the audio path never de-identified anything.

    grep -n "scrub_pii\|find_ambiguities" mao/agents/multimodal_agent.py

returned NOTHING. `handle_audio` transcribed the upload and concatenated the raw
Whisper output straight into the user turn, and returned `transcript[:500]` in
`result_meta` — which is echoed into the response, cached in Redis under two
keys and written to a trace. A recorded consultation contains spoken names,
addresses and dates of birth by construction, and `audio_b64` is the documented
way to send one. It is reachable: `clinical_agent`'s `elif has_audio:` branch,
which every attachment is routed to.

Like ADV15-8, this was never a de-identification defect. The de-identifier was
not called. `01_ARCHITECTURE.md`: "any existing audio/transcript path must obey
the same protected-input contract or be disabled."

`import whisper` raises `ModuleNotFoundError` on this host, so transcription is
stubbed rather than depended on — which is also the only way to make the
transcript a fixed, assertable string. The stub replaces the *transcriber*; the
boundary, the agent, the graph and the egress gateway all run their production
code, and the recorder is bound at `gateway.set_provider`, the real sink.
"""
from __future__ import annotations

import base64
import sys
import types

import pytest
from fastapi.testclient import TestClient

from mao.providers import gateway
from mao.providers.llm.base import ProviderResponse
from tests.trust.recorders import RecordingProvider

# Synthetic. Invented for this fixture.
PATIENT_NAME = "Harold Nkemdirim"
MRN = "RGT/44219/B"
NHS_NUMBER = "943 476 5919"
DOB = "12/03/1948"

PHI = [PATIENT_NAME, MRN, NHS_NUMBER, DOB]

# What a dictated clinic recording transcribes to.
TRANSCRIPT = (
    f"Patient Name: {PATIENT_NAME}, MRN: {MRN}, NHS Number: {NHS_NUMBER}, "
    f"DOB: {DOB}. He reports worsening short-term memory over six months "
    "and was started on donepezil 5 mg once daily."
)

# A spoken header whose end cannot be established — dictation has no
# punctuation, so this is the ordinary case, not a contrived one.
AMBIGUOUS_TRANSCRIPT = (
    "Patient: Gordon Whitfield Rockwood Frailty Scale 5 was recorded today"
)

AUDIO_B64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode()


def _install_fake_whisper(monkeypatch: pytest.MonkeyPatch, transcript: str) -> None:
    """Replace the transcriber, nothing else.

    `import whisper` inside `handle_audio` resolves from `sys.modules` first, so
    this is the seam that stands in for a model that is not installed here —
    and it leaves every control under test running its real implementation.
    """

    class _Model:
        def transcribe(self, path: str) -> dict:
            return {"text": transcript}

    module = types.ModuleType("whisper")
    module.load_model = lambda name: _Model()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", module)


class _PlausibleRecorder(RecordingProvider):
    """Records at the real sink; answers in the shape each caller parses."""

    def complete(self, *, model_id, messages, temperature, max_tokens):
        super().complete(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
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
            text = "The recording describes a cholinesterase inhibitor being started."
        return ProviderResponse(text=text, input_tokens=10, output_tokens=10)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch):
    from mao.agents import clinical_agent
    from mao.memory.interface import reset_memory_store, set_memory_store
    from mao.safety import verification
    from tests.trust.recorders import RecordingMemory

    provider = _PlausibleRecorder()
    gateway.set_provider(provider)
    memory = RecordingMemory()
    set_memory_store(memory)
    provider.memory = memory

    monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
    monkeypatch.setattr(verification, "check_all_claims", lambda claims, premise: [])

    yield provider

    gateway.reset_provider()
    reset_memory_store()


@pytest.fixture
def client(recorder):
    from mao.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _post(client: TestClient, route: str = "/chat"):
    return client.post(
        route,
        json={
            "query": "What did the patient report in this recording?",
            "user_id": "audio-probe",
            "metadata": {"audio_b64": AUDIO_B64},
        },
    )


class TestTheTranscriptEntersTheProtectedBoundary:
    @pytest.fixture(autouse=True)
    def _whisper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_whisper(monkeypatch, TRANSCRIPT)

    def test_the_request_succeeds(self, client, recorder) -> None:
        assert _post(client).status_code == 200, _post(client).text

    def test_no_identifier_from_the_transcript_reaches_the_provider(
        self, client, recorder
    ) -> None:
        _post(client)
        assert recorder.calls, "the provider was never called — the probe is vacuous"
        assert recorder.leaked(PHI) == [], (
            "these identifiers from the audio transcript were sent to the provider"
        )

    def test_the_clinical_content_still_reaches_the_provider(
        self, client, recorder
    ) -> None:
        """Or the assertion above passes because nothing was transcribed at all."""
        _post(client)
        assert recorder.calls, "the provider was never called — the probe is vacuous"
        assert "donepezil" in recorder.text()

    def test_no_raw_transcript_is_returned_to_the_caller(
        self, client, recorder
    ) -> None:
        """`result_meta["transcript"][:500]` was the first 500 characters of a
        consultation, which is exactly where the spoken header is. It is echoed
        into the response body, cached under two Redis keys and traced."""
        body = _post(client).json()
        blob = repr(body)
        leaked = [identifier for identifier in PHI if identifier in blob]
        assert leaked == [], f"the response carried {leaked}"
        assert "transcript" not in body["metadata"], (
            "raw transcript text is back in the response metadata"
        )

    def test_the_transcript_length_is_still_reported(self, client, recorder) -> None:
        """The provenance survives the strip — a caller can still tell that a
        recording was transcribed and roughly how much of it there was."""
        metadata = _post(client).json()["metadata"]
        assert metadata["transcript_chars"] > 0
        assert metadata["mode"] == "audio"

    def test_the_memory_store_receives_no_identifier(self, client, recorder) -> None:
        _post(client)
        assert recorder.memory.calls, "nothing reached memory — the probe is vacuous"
        assert recorder.memory.leaked(PHI) == []


class TestTheProbeCanFail:
    """Mutation control — remove the boundary call, expect the leak back.

    Bound at the boundary's scrub rather than at `protect_channel` itself, so
    the control mirrors the history probe's and both fail for the same reason.
    """

    @pytest.fixture(autouse=True)
    def _whisper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_whisper(monkeypatch, TRANSCRIPT)

    def test_every_identifier_leaks_without_the_boundary(
        self, client, recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mao.core.deident.report import ScrubResult
        from mao.trust.inputs import boundary

        monkeypatch.setattr(
            boundary, "scrub_with_report", lambda text: ScrubResult(text=text)
        )
        _post(client)
        assert recorder.calls, "the provider was never called — the probe is vacuous"
        assert sorted(recorder.leaked(PHI)) == sorted(PHI), (
            "disabling the scrub did not reopen the leak, so this probe is not "
            "bound to the seam that carries the transcript"
        )


class TestAnAmbiguousSpokenHeaderIsRefused:
    """A recording is processed unseen for the same reason a PDF is.

    Dictation has no punctuation, so "the end of the name cannot be established"
    is the ordinary case for speech rather than a contrived one. The upload
    posture applies: refuse and ask for structured fields, on both routes.
    """

    @pytest.fixture(autouse=True)
    def _whisper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_whisper(monkeypatch, AMBIGUOUS_TRANSCRIPT)

    @pytest.mark.parametrize("route", ["/chat", "/chat/stream"])
    def test_the_route_answers_422(self, client, recorder, route: str) -> None:
        response = _post(client, route)
        assert response.status_code == 422, response.text
        assert "structured metadata fields" in response.json()["detail"]

    @pytest.mark.parametrize("route", ["/chat", "/chat/stream"])
    def test_the_refusal_quotes_no_spoken_content(
        self, client, recorder, route: str
    ) -> None:
        detail = _post(client, route).json()["detail"]
        assert "Whitfield" not in detail
        assert "Rockwood" not in detail
