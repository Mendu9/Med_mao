r"""C5 and C6, under control decision M-2.

AUDIO - A-6 and ADV16-5. Wave 12 made `handle_audio` call `protect_channel`,
which is the right call signature. ADV16-5 then stubbed `whisper` into
`sys.modules` so the real body ran end to end against a realistic dictation and
measured what that signature actually delivers on continuous speech: 5 of 7
identifiers reach the provider - patient name, hospital number in spoken form
("RGT slash 44219 slash B"), NHS number, date of birth and telephone all
spelled out in words. The ONE name redacted is the CLINICIAN's, because the
title "Doctor" precedes it. The PATIENT's name, the one the invariant exists
for, is the one that survives. A-6 is the other half: `handle_audio` never
passed structured patient fields, so a caller doing exactly what the 422 asks
got the same 422 again - the refusal was a wall, not a request.

`01_ARCHITECTURE.md` requires an audio path to "obey the same protected-input
contract OR BE DISABLED". Closing this any other way needs a spoken-identifier
matcher, and the control plane has declined to build one.

IMAGE - A-5 and ADV16-7, which both reviews reached independently. The vision
call declared no trust class and so took `complete()`'s default of
SAFE_DERIVED_TEXT - "de-identified text minted by the protected input boundary"
- for a base64 patient scan that had been through no boundary at all.

Scope, stated precisely: what is disabled is EXTERNAL EGRESS of raw imagery.
The local EfficientNetB3 stage predictor runs in-process, sends nothing
anywhere, and is untouched.

The boundary that failed for both is the PROVIDER SEAM, so that is where these
assertions are bound. "It returns a refusal string" is satisfiable by a
function that also made the call.
"""
from __future__ import annotations

import base64
import sys
import types

import pytest


@pytest.fixture()
def provider_calls(monkeypatch):
    """Fails the test from inside if a disabled modality reaches the gateway."""
    calls: list[dict] = []

    def _record(**kwargs):
        calls.append(kwargs)
        raise AssertionError("a disabled modality reached the provider")

    from mao.providers import gateway

    monkeypatch.setattr(gateway, "complete", _record)
    return calls


@pytest.fixture()
def stub_whisper(monkeypatch):
    """ADV16-5's method. `whisper` is not installed on this host, so without
    this the function returns early for the wrong reason and every assertion
    below would pass vacuously."""
    transcribed: list[str] = []
    module = types.ModuleType("whisper")

    class _Model:
        @staticmethod
        def transcribe(path: str) -> dict:
            transcribed.append(path)
            return {
                "text": "Good morning, this is Doctor Alan Rankin dictating. "
                "The patient is Harold Nkemdirim, hospital number RGT slash "
                "44219 slash B, NHS number nine four three, four seven six."
            }

    module.load_model = lambda name: _Model()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", module)
    return transcribed


_AUDIO = base64.b64encode(b"RIFF0000WAVEfmt ").decode()
_SCAN = base64.b64encode(b"\xff\xd8\xff\xe0JFIF Harold Nkemdirim RGT/44219/B").decode()


class TestTheAudioPathIsDisabled:
    def test_it_refuses_without_transcribing_or_calling_a_provider(
        self, provider_calls, stub_whisper
    ) -> None:
        from mao.agents.multimodal_agent import handle_audio

        _, meta = handle_audio("Summarise this consultation.", {"audio_b64": _AUDIO}, "")

        assert meta.get("error") == "audio_disabled_phase_1"
        assert provider_calls == []
        assert stub_whisper == [], (
            "the recording was transcribed. A disabled path must not create a "
            "transcript it then has to protect, log or hold in memory."
        )

    def test_the_refusal_does_not_promise_a_remedy_this_channel_lacks(self) -> None:
        """A-6. The report path's 422 asks for structured patient fields and
        resolving it works there. On the transcript channel `protect_channel`
        was never passed them, so repeating that instruction here would be the
        same wall wearing a helpful sentence."""
        from mao.agents.multimodal_agent import handle_audio

        answer, _ = handle_audio("q", {"audio_b64": _AUDIO}, "")
        assert "disabled" in answer.lower()
        assert "structured" not in answer.lower()

    def test_the_refusal_does_not_echo_the_caller_payload(self) -> None:
        from mao.agents.multimodal_agent import handle_audio

        answer, meta = handle_audio("q", {"audio_b64": _AUDIO}, "")
        assert _AUDIO not in answer
        assert _AUDIO not in str(meta)

    def test_the_gate_is_what_stops_it_and_not_whispers_absence(
        self, stub_whisper, monkeypatch
    ) -> None:
        """Mutation control. With the gate lifted the stub IS reached, so the
        first test measures the gate rather than a missing dependency."""
        import mao.agents.multimodal_agent as module

        monkeypatch.setattr(module, "AUDIO_ENABLED", True)
        monkeypatch.setattr(
            module.gateway,
            "complete",
            lambda **k: types.SimpleNamespace(text="ok", model_id="m"),
        )
        module.handle_audio("q", {"audio_b64": _AUDIO}, "")
        assert stub_whisper, "the gate is not what stopped the transcription"


class TestRawImageryDoesNotLeaveTheProcess:
    def test_handle_image_refuses_without_calling_a_provider(
        self, provider_calls
    ) -> None:
        from mao.agents.multimodal_agent import handle_image

        answer, meta = handle_image(
            "Does this scan show atrophy?", {"image_b64": _SCAN}, ""
        )

        assert meta.get("error") == "image_disabled_phase_1"
        assert provider_calls == []

    def test_the_refusal_carries_neither_the_image_nor_an_identifier(self) -> None:
        from mao.agents.multimodal_agent import handle_image

        answer, meta = handle_image("q", {"image_b64": _SCAN}, "")
        assert _SCAN not in answer and _SCAN not in str(meta)
        assert "Nkemdirim" not in answer

    def test_it_does_not_fetch_a_caller_supplied_image_url_either(
        self, monkeypatch, provider_calls
    ) -> None:
        """A disabled egress must not first pull the image across the network
        to hold it in memory. The refusal comes before any I/O at all."""
        fetched: list[str] = []
        import mao.safety.fetch as fetch_module

        monkeypatch.setattr(
            fetch_module,
            "fetch_image_bytes",
            lambda url: fetched.append(url) or b"x",
        )
        from mao.agents.multimodal_agent import handle_image

        handle_image("q", {"image_url": "https://example.invalid/scan.jpg"}, "")
        assert fetched == []

    def test_the_policy_table_refuses_the_flow_independently(self) -> None:
        """Wall two. Even if a call site reappears, the flow is unnamed and
        `authorise` rejects it before any trust class is considered."""
        from mao.trust.classes import TrustClass
        from mao.trust.egress.gateway import EgressRefused, authorise
        from mao.trust.egress.policy import Destination, EgressPurpose

        with pytest.raises(EgressRefused, match="no approved flow"):
            authorise(
                destination=Destination.MODEL_PROVIDER,
                purpose=EgressPurpose.IMAGE_ANALYSIS,
                trust_class=TrustClass.SAFE_DERIVED_TEXT,
            )

    def test_the_local_stage_predictor_is_not_what_was_disabled(self) -> None:
        """Scope check, asserted so a later reader does not widen M-2 into
        "no image support". EfficientNetB3 runs in-process and sends nothing
        anywhere; it is not an egress and it is untouched."""
        from mao.agents import clinical_agent

        assert hasattr(clinical_agent, "_run_mri_prediction")


class TestTheClinicalRouteStillAnswersWithAnAttachment:
    """A disabled modality must degrade honestly, not crash and not silently
    answer as though nothing was attached - which is the defect
    `clinical_agent`'s own comment records for the audio branch."""

    def test_an_audio_upload_gets_an_answer_that_says_what_happened(
        self, stub_whisper
    ) -> None:
        from mao.agents.clinical_agent import clinical_node

        state = clinical_node(
            {
                "user_query": "Summarise this consultation.",
                "metadata": {"audio_b64": _AUDIO},
                "memory_context": "",
            }
        )
        assert state["response"].strip()
        assert "disabled" in state["response"].lower()
