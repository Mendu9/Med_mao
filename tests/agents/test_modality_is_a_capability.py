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

The second class below was found while checking that `_handle_mri_image` never
looked at the image when MRI prediction failed: a non-MRI upload — a photo of a
rash, a pill bottle, an ECG — took the "prediction unavailable" branch and was
answered from retrieval alone, with no indication the picture had been ignored.
That workflow is retired under M-3, but the silent-drop invariant it was written
for is not, so the class now binds to how the image branch behaves today.
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
    """The invariant survives M-3; what satisfies it changed.

    These tests used to drive `_handle_mri_image`: when stage prediction
    failed, the image had to be described by the vision model rather than
    answered from retrieval alone. That whole path is retired — the predictor
    is deleted and the vision egress is shut by M-2 — so there is nothing left
    to fall back BETWEEN.

    What must still hold is the thing the class is named for: an uploaded image
    must never be quietly dropped so the caption gets answered as though
    nothing was attached. It is now satisfied by refusing out loud.
    """

    def test_an_image_upload_is_answered_by_saying_the_modality_is_unavailable(
        self,
    ) -> None:
        state = clinical_agent.clinical_node(
            {
                "user_query": "Does this scan show atrophy?",
                "metadata": {"image_b64": "eA=="},
                "memory_context": "",
            }
        )
        response = state["response"].lower()
        assert response.strip()
        assert "image analysis is not available" in response, (
            "the upload was answered without saying the image was not used"
        )

    def test_the_image_branch_is_what_answers_and_not_the_text_path(self) -> None:
        """Non-vacuity control. If the `has_image` branch were deleted rather
        than repointed, the request would fall through to `_handle_text_question`
        and this mode would read `text_question` — which is precisely the silent
        drop this class exists to prevent."""
        state = clinical_agent.clinical_node(
            {
                "user_query": "Does this scan show atrophy?",
                "metadata": {"image_b64": "eA=="},
                "memory_context": "",
            }
        )
        assert state["metadata"].get("mode") != "text_question"

    def test_the_retired_predictor_entrypoints_are_gone(self) -> None:
        """M-3. Named individually so a partial revert fails loudly here."""
        for attribute in (
            "_handle_mri_image",
            "_run_mri_prediction",
            "_format_prediction",
            "_interpret_stage",
            "_describe_image",
        ):
            assert not hasattr(clinical_agent, attribute), (
                f"clinical_agent.{attribute} survived the M-3 retirement"
            )
