"""Scope item 12 — modality handled as a capability, not a top-level agent.

`01_ARCHITECTURE.md`: "avoid top-level Multimodal Agent when modality can be
handled as workflow capability." The architecture review recorded this as
PARTIAL because both the Tool agent and the Multimodal agent survived.

Re-investigating showed the multimodal node was not merely redundant but
unreachable-with-media: `router_node` forces *every* attachment to `clinical`,
so `multimodal_node` could only ever be entered with no attachment at all — in
which case its only possible reply is "please provide an image or audio". A
top-level agent that cannot receive the data it exists to process is not a
design choice.

Found while checking that: `_handle_mri_image` never looks at the image when MRI
prediction fails. A non-MRI upload — a photo of a rash, a pill bottle, an ECG —
takes the "prediction unavailable" branch and is answered from retrieval alone,
with no indication the picture was ignored. Same silent-drop class as the audio
finding.
"""
from __future__ import annotations

import pytest

from mao.agents import clinical_agent


class TestModalityIsNotATopLevelAgent:
    def test_the_graph_has_no_multimodal_node(self) -> None:
        from mao.graph import build_graph

        assert "multimodal_node" not in set(build_graph().nodes)

    def test_the_multimodal_intent_routes_to_a_capable_node(self) -> None:
        from mao.agents.router import route_to_agent

        assert route_to_agent({"intent": "multimodal"}) == "clinical_node"

    def test_the_multimodal_intent_is_still_a_known_intent(self) -> None:
        """Removing the node must not make a classifier label unroutable."""
        from mao.agents.router import route_to_agent
        from mao.core.state import ALL_INTENTS

        assert "multimodal" in ALL_INTENTS
        assert route_to_agent({"intent": "multimodal"})

    @pytest.mark.parametrize("key", ["image_b64", "report_b64", "audio_b64"])
    def test_every_attachment_still_reaches_a_clinical_route(self, key: str) -> None:
        from mao.agents.router import route_to_agent, router_node

        state = router_node(
            {"user_query": "what is this?", "user_id": "u", "metadata": {key: "x"}}
        )
        assert route_to_agent(state) == "clinical_node"

    def test_the_capability_functions_survive(self) -> None:
        """Removing the node must not remove the vision and audio capabilities."""
        from mao.agents import multimodal_agent

        assert callable(multimodal_agent.handle_audio)
        assert callable(multimodal_agent.handle_image)


class TestAnImageIsNeverSilentlyIgnored:
    def test_a_failed_mri_prediction_falls_back_to_vision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            clinical_agent,
            "_run_mri_prediction",
            lambda metadata: {"error": "not an MRI"},
        )
        monkeypatch.setattr(
            clinical_agent,
            "handle_image",
            lambda *a, **k: ("A photograph of a pill bottle.", {"vision_model": "stub"}),
        )
        monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
        monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")

        seen: dict[str, str] = {}

        def _call_llm(system_prompt: str, user_prompt: str) -> str:
            seen["user"] = user_prompt
            return "answer"

        monkeypatch.setattr(clinical_agent, "_call_llm", _call_llm)

        clinical_agent._handle_mri_image("what is this?", {"image_b64": "x"}, "")
        assert "pill bottle" in seen["user"], (
            "the image was never described to the model"
        )

    def test_a_successful_mri_prediction_does_not_pay_for_vision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            clinical_agent,
            "_run_mri_prediction",
            lambda metadata: {
                "prediction": "AD",
                "full_label": "Alzheimer's disease",
                "confidence": 0.91,
                "all_scores": {"AD": 0.91},
            },
        )
        monkeypatch.setattr(
            clinical_agent,
            "handle_image",
            lambda *a, **k: (called.append("vision"), ("x", {}))[1],
        )
        monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
        monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
        monkeypatch.setattr(clinical_agent, "_call_llm", lambda s, u: "answer")

        clinical_agent._handle_mri_image("stage this scan", {"image_b64": "x"}, "")
        assert called == []

    def test_the_mode_records_that_vision_was_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            clinical_agent, "_run_mri_prediction", lambda metadata: {"error": "no"}
        )
        monkeypatch.setattr(
            clinical_agent, "handle_image", lambda *a, **k: ("a rash", {})
        )
        monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
        monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
        monkeypatch.setattr(clinical_agent, "_call_llm", lambda s, u: "answer")

        _, meta = clinical_agent._handle_mri_image("what is this?", {"image_b64": "x"}, "")
        assert meta.get("vision_fallback") is True

    def test_a_vision_failure_does_not_break_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            clinical_agent, "_run_mri_prediction", lambda metadata: {"error": "no"}
        )

        def _boom(*a: object, **k: object) -> tuple[str, dict]:
            raise RuntimeError("vision down")

        monkeypatch.setattr(clinical_agent, "handle_image", _boom)
        monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
        monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
        monkeypatch.setattr(clinical_agent, "_call_llm", lambda s, u: "answer")

        response, _ = clinical_agent._handle_mri_image("what?", {"image_b64": "x"}, "")
        assert response == "answer"
