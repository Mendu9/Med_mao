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


class TestTheAudioPathIsDisabledAndNothingLeaves:
    """Control decision M-2: the external audio path is DISABLED for Phase 1.

    Wave 12 closed ADV15-15's STRUCTURAL half - the transcript did enter
    `protect_channel`, and the raw transcript did stop being echoed back. The
    b63311d adversarial review (ADV16-5) then executed the real body against a
    realistic CONTINUOUS-SPEECH dictation rather than the labelled banner this
    file's TRANSCRIPT fixture uses, and measured what that boundary actually
    delivers on speech: 5 of 7 identifiers reach the provider. The patient's
    name survives; the clinician's is redacted, because the title "Doctor"
    precedes it.

    This fixture is part of why the gap went unseen here. `TRANSCRIPT` above is
    written with `Patient Name:` / `MRN:` / `NHS Number:` labels, which is what
    the labelled-value machinery is built for - but Whisper emits continuous
    prose with no labels and no digit groups, so the shape this file tested is
    not the shape the modality produces. It is kept verbatim rather than
    rewritten, because it is the record of how a green suite and a real leak
    coexisted.

    What these assertions establish now is stronger and narrower: nothing
    reaches the provider, nothing reaches memory, and no recording is
    transcribed at all.
    """

    @pytest.fixture(autouse=True)
    def _whisper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_whisper(monkeypatch, TRANSCRIPT)

    def test_the_request_still_succeeds(self, client, recorder) -> None:
        """A disabled modality degrades honestly. It does not 500."""
        assert _post(client).status_code == 200, _post(client).text

    def test_the_caller_is_told_the_recording_was_not_processed(
        self, client, recorder
    ) -> None:
        """`clinical_agent` routes audio deliberately because "silently
        dropping it is a clinical failure". Disabling it must not reintroduce
        that silence: the answer has to say the recording was not used."""
        body = _post(client).json()
        assert "disabled" in body["response"].lower()

    def test_no_identifier_from_the_transcript_reaches_the_provider(
        self, client, recorder
    ) -> None:
        _post(client)
        assert recorder.leaked(PHI) == [], (
            "these identifiers from the audio transcript were sent to the provider"
        )

    def test_nothing_was_transcribed_at_all(self, client, recorder) -> None:
        """The reason the assertion above holds, stated separately so the two
        cannot be confused. A disabled path must not create a transcript it
        would then have to protect, cache, log or hold in memory - which is a
        stronger property than de-identifying one."""
        body = _post(client).json()
        assert body["metadata"].get("transcript_chars", 0) == 0
        assert "transcript" not in body["metadata"]

    def test_no_raw_transcript_is_returned_to_the_caller(
        self, client, recorder
    ) -> None:
        """`result_meta["transcript"][:500]` was the first 500 characters of a
        consultation, which is exactly where the spoken header is. It was
        echoed into the response body, cached under two Redis keys and traced."""
        blob = repr(_post(client).json())
        leaked = [identifier for identifier in PHI if identifier in blob]
        assert leaked == [], f"the response carried {leaked}"

    def test_the_memory_store_receives_no_identifier(self, client, recorder) -> None:
        _post(client)
        assert recorder.memory.leaked(PHI) == []

    def test_the_mode_is_still_reported_so_the_drop_is_visible(
        self, client, recorder
    ) -> None:
        """Was `transcript_chars > 0` and `mode == "audio"`. The length is gone
        because there is no transcript; the mode stays, so a caller can still
        tell that an audio upload was recognised and deliberately not processed
        rather than silently ignored. The length assertion is owed back by the
        re-enable gate."""
        assert _post(client).json()["metadata"]["mode"] == "audio"


class TestWhatTheGateIsHoldingBack:
    """ADV16-5, reproduced in the suite, as the mutation control.

    With the gate lifted the real body runs and a realistic dictation leaks.
    This is what makes the class above non-vacuous: it proves those assertions
    pass because of `AUDIO_ENABLED`, and not because `whisper` is absent from
    this host or because the fixture transcribes to nothing.

    It is also the standing record of what the Voice gate owes. If someone
    flips `AUDIO_ENABLED` to True without building a transformation for
    continuous speech, this test fails and says why.
    """

    #: Continuous speech, as Whisper actually emits it: no field labels, no
    #: digit groups, identifiers spelled out in words. The reviewer's.
    SPOKEN = (
        "Good morning, this is Doctor Alan Rankin dictating. The patient is "
        f"{PATIENT_NAME}, hospital number RGT slash 44219 slash B, NHS number "
        "nine four three, four seven six, five nine one nine, date of birth "
        "twelve oh three nineteen forty eight. He has probable Alzheimer "
        "disease and was started on donepezil 5 mg once daily."
    )

    @pytest.fixture(autouse=True)
    def _whisper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_whisper(monkeypatch, self.SPOKEN)

    def test_with_the_gate_lifted_the_patient_name_reaches_the_provider(
        self, client, recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mao.agents import multimodal_agent

        monkeypatch.setattr(multimodal_agent, "AUDIO_ENABLED", True)
        _post(client)

        assert recorder.calls, "the provider was never called - probe is vacuous"
        assert PATIENT_NAME in recorder.text(), (
            "the spoken patient name did NOT leak with the gate lifted, so "
            "either the modality is now safe - in which case reconsider M-2 "
            "with evidence - or this probe is no longer bound to the seam that "
            "carries the transcript"
        )

    def test_the_clinicians_name_is_the_one_that_gets_redacted(
        self, client, recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sharpest part of ADV16-5, kept because it explains why a
        vocabulary fix cannot close this: the boundary redacts the name with a
        title in front of it and keeps the one without."""
        from mao.agents import multimodal_agent

        monkeypatch.setattr(multimodal_agent, "AUDIO_ENABLED", True)
        _post(client)
        blob = recorder.text()
        assert "Alan Rankin" not in blob
        assert PATIENT_NAME in blob

    def test_the_gate_and_not_the_scrubber_is_what_stops_it(
        self, client, recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The original mutation control, re-aimed. Disabling the scrub while
        the gate is DOWN must change nothing, because nothing is transcribed."""
        from mao.core.deident.report import ScrubResult
        from mao.trust.inputs import boundary

        monkeypatch.setattr(
            boundary, "scrub_with_report", lambda text: ScrubResult(text=text)
        )
        _post(client)
        assert recorder.leaked(PHI) == []


class TestAnAmbiguousSpokenHeaderNoLongerNeedsARefusal:
    """A-6. The 422 this used to assert was a wall, not a request.

    The report path resolves an ambiguous header from caller-stated fields:
    `clinical_agent` passes `structured=_structured_fields(metadata)`. The
    audio path called `protect_channel(transcript, InputChannel.TRANSCRIPT)`
    with no `structured` argument at all, so a caller who did exactly what the
    422 asked - "Send the patient identifiers as structured metadata fields" -
    got the same 422 again. There was no audio equivalent of
    `test_supplying_the_fields_makes_the_same_document_process`, and there
    could not have been one.

    Under M-2 there is nothing to refuse: the recording is not processed. The
    re-enable gate owes BOTH the structured-field pathway and a transformation
    for continuous speech.
    """

    @pytest.fixture(autouse=True)
    def _whisper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_whisper(monkeypatch, AMBIGUOUS_TRANSCRIPT)

    @pytest.mark.parametrize("route", ["/chat", "/chat/stream"])
    def test_the_route_answers_200_and_says_the_recording_was_not_used(
        self, client, recorder, route: str
    ) -> None:
        response = _post(client, route)
        assert response.status_code == 200, response.text
        assert "disabled" in response.text.lower()

    @pytest.mark.parametrize("route", ["/chat", "/chat/stream"])
    def test_the_answer_quotes_no_spoken_content(
        self, client, recorder, route: str
    ) -> None:
        """Unchanged in force: whatever the route says about a recording, it
        must not repeat what was in it."""
        body = _post(client, route).text
        assert "Whitfield" not in body
        assert "Rockwood" not in body
