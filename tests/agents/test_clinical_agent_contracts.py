"""Clinical-agent findings from both exit-gate reviews.

arch-H1 — `clinical_agent` never imported `mao.prompts`. It ran an inline
`_CLINICAL_SYSTEM`, and the registered `clinical.extraction` prompt had already
diverged: the registry copy carries "The report has already been de-identified;
do not attempt to infer patient identity." and the inline copy does not. The
prompt test asserted *registration*, not *consumption*, so the registry stayed
green while the highest-risk agent in the system ran unregistered, untraceable,
divergent prompt text.

adv-M2 — the router forces *any* attachment to `clinical`, but `clinical_node`
branched only on image/report, so uploaded audio was accepted, priced as HIGH
risk, and then silently discarded. A clinician uploading a voice sample — a
recognised Alzheimer's biomarker modality — got a generic text answer with no
indication the audio was dropped.

adv-M3 — `_RAW_ATTACHMENT_KEYS` stripped only the two base64 keys, so
`report_path`, `image_url`, `audio_b64` and `audio_path` survived into the
returned and cached metadata. Filenames routinely carry PHI
(`/uploads/JohnSmith_MRN12345.pdf`).
"""
from __future__ import annotations

import inspect

import pytest

from mao.agents import clinical_agent
from mao.prompts import get_prompt
from mao.safety.policy import ATTACHMENT_KEYS


class TestPromptsComeFromTheRegistry:
    def test_the_agent_imports_the_prompt_registry(self) -> None:
        assert "mao.prompts" in inspect.getsource(clinical_agent)

    def test_no_inline_system_prompt_constants_survive(self) -> None:
        assert not hasattr(clinical_agent, "_CLINICAL_SYSTEM")
        assert not hasattr(clinical_agent, "_EXTRACTION_SYSTEM")

    def test_the_synthesis_prompt_is_the_registered_one(self) -> None:
        assert clinical_agent._clinical_system() == get_prompt("clinical.synthesis").template

    def test_the_extraction_prompt_is_the_registered_one(self) -> None:
        assert clinical_agent._extraction_system() == get_prompt("clinical.extraction").template

    def test_the_de_identification_instruction_actually_reaches_the_model(self) -> None:
        """The exact divergence the architecture review found."""
        assert "already been de-identified" in clinical_agent._extraction_system()


class TestAttachmentIdentifiersDoNotSurvive:
    @pytest.mark.parametrize("key", sorted(ATTACHMENT_KEYS))
    def test_no_attachment_key_survives_into_returned_metadata(
        self, key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            clinical_agent,
            "_handle_text_question",
            lambda *a, **k: ("answer", {}),
        )
        state = {
            "user_query": "what does this show?",
            "metadata": {key: "/uploads/JohnSmith_MRN12345.pdf"},
        }
        out = clinical_agent.clinical_node(state)
        assert key not in out["metadata"]

    def test_unrelated_metadata_is_still_echoed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            clinical_agent,
            "_handle_text_question",
            lambda *a, **k: ("answer", {}),
        )
        out = clinical_agent.clinical_node(
            {"user_query": "q", "metadata": {"locale": "en-GB"}}
        )
        assert out["metadata"]["locale"] == "en-GB"

    def test_the_agent_does_not_keep_its_own_key_list(self) -> None:
        source = inspect.getsource(clinical_agent)
        assert '"image_b64", "report_b64"' not in source
        assert "ATTACHMENT_KEYS" in source


class TestAudioIsNotSilentlyDropped:
    def test_audio_reaches_a_handler_rather_than_the_text_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: dict[str, bool] = {"audio": False, "text": False}

        def _audio(*a: object, **k: object) -> tuple[str, dict]:
            called["audio"] = True
            return ("transcript-based answer", {"mode": "audio"})

        def _text(*a: object, **k: object) -> tuple[str, dict]:
            called["text"] = True
            return ("generic answer", {"mode": "text_question"})

        monkeypatch.setattr(clinical_agent, "handle_audio", _audio)
        monkeypatch.setattr(clinical_agent, "_handle_text_question", _text)

        clinical_agent.clinical_node(
            {"user_query": "does this suggest decline?", "metadata": {"audio_b64": "<wav>"}}
        )
        assert called["audio"] is True
        assert called["text"] is False

    def test_the_mode_reports_audio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            clinical_agent,
            "handle_audio",
            lambda *a, **k: ("answer", {"mode": "audio"}),
        )
        out = clinical_agent.clinical_node(
            {"user_query": "q", "metadata": {"audio_path": "/tmp/a.wav"}}
        )
        assert out["metadata"]["mode"] == "audio"

    def test_audio_path_also_routes_to_audio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr(
            clinical_agent,
            "handle_audio",
            lambda *a, **k: (seen.append("audio"), ("a", {"mode": "audio"}))[1],
        )
        clinical_agent.clinical_node(
            {"user_query": "q", "metadata": {"audio_path": "/tmp/a.wav"}}
        )
        assert seen == ["audio"]

    def test_image_still_wins_over_audio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An MRI is the stronger clinical signal when both are present."""
        seen: list[str] = []
        monkeypatch.setattr(
            clinical_agent,
            "_handle_mri_image",
            lambda *a, **k: (seen.append("image"), ("a", {"mode": "mri_image"}))[1],
        )
        monkeypatch.setattr(
            clinical_agent,
            "handle_audio",
            lambda *a, **k: (seen.append("audio"), ("a", {}))[1],
        )
        clinical_agent.clinical_node(
            {"user_query": "q", "metadata": {"image_b64": "x", "audio_b64": "y"}}
        )
        assert seen == ["image"]

    def test_there_is_one_shared_audio_implementation(self) -> None:
        from mao.agents import multimodal_agent

        assert clinical_agent.handle_audio is multimodal_agent.handle_audio


class TestTheDisclaimerStillSurvives:
    def test_every_clinical_response_carries_the_disclaimer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mao.safety.verification import DISCLAIMER_MARKER

        monkeypatch.setattr(
            clinical_agent, "_handle_text_question", lambda *a, **k: ("answer", {})
        )
        out = clinical_agent.clinical_node({"user_query": "q", "metadata": {}})
        assert DISCLAIMER_MARKER in out["response"]
