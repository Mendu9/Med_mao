"""L1 — a raw attachment payload must never reach graph state.

`state["metadata"]` is persisted, traced, logged and cached, so spreading the
inbound metadata into it carried the raw base64 scan or voice sample along for
the ride.

These tests used to drive `multimodal_node`. That node was removed in Wave 7 as
unreachable-with-media — `router_node` forces every attachment to `clinical`, so
it could only ever be entered with nothing attached. The property it was
protecting did not go away with it; modality is now a capability of
`clinical_node`, which is where every attachment actually lands, so that is
where the property is asserted.

`test_non_attachment_metadata_is_preserved` is gone rather than ported. It
asserted that caller metadata outside `ATTACHMENT_KEYS` survives into the
response, which is adversarial finding H-2 stated as a requirement — the rule
that let `Doe_Jane_MRN4471023_1948-03-12.pdf` reach Redis. Its replacement
asserts the opposite, and the full contract lives in
tests/api/test_caller_metadata_is_not_trusted.py.
"""
from __future__ import annotations

import pytest

from mao.safety.policy import ATTACHMENT_KEYS


@pytest.fixture
def run_clinical(monkeypatch: pytest.MonkeyPatch):
    from mao.agents import clinical_agent

    monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
    monkeypatch.setattr(clinical_agent, "_call_llm", lambda s, u: "A description.")
    monkeypatch.setattr(
        clinical_agent,
        "_run_mri_prediction",
        lambda metadata: {"error": "not an MRI"},
    )
    monkeypatch.setattr(
        clinical_agent,
        "handle_image",
        lambda *a, **k: ("A description of the image.", {"vision_model": "stub"}),
    )
    monkeypatch.setattr(
        clinical_agent,
        "handle_audio",
        lambda *a, **k: ("A transcript summary.", {"transcript": "hello"}),
    )

    def _run(metadata: dict) -> dict:
        state = {
            "user_query": "what does this show?",
            "user_id": "u1",
            "metadata": metadata,
            "memory_context": "",
        }
        return dict(clinical_agent.clinical_node(state))

    return _run


class TestNoRawPayloadInState:
    def test_image_payload_is_not_echoed(self, run_clinical) -> None:
        out = run_clinical({"image_b64": "SENSITIVE_SCAN_BYTES", "modality": "image"})
        assert "SENSITIVE_SCAN_BYTES" not in str(out["metadata"])

    def test_audio_payload_is_not_echoed(self, run_clinical) -> None:
        out = run_clinical({"audio_b64": "SENSITIVE_VOICE_BYTES", "modality": "audio"})
        assert "SENSITIVE_VOICE_BYTES" not in str(out["metadata"])

    def test_no_attachment_key_survives_into_state(self, run_clinical) -> None:
        out = run_clinical({"image_b64": "x", "audio_b64": "y", "modality": "image"})
        for key in ATTACHMENT_KEYS:
            assert key not in out["metadata"], f"{key} leaked into graph state"

    def test_no_caller_metadata_survives_into_state(self, run_clinical) -> None:
        """Inverted — see the module docstring. `case_id` is as identifying as
        the filename that provoked H-2."""
        out = run_clinical(
            {"image_b64": "x", "modality": "image", "case_id": "abc-123"}
        )
        assert "case_id" not in out["metadata"]
        assert "abc-123" not in str(out["metadata"])

    def test_the_agents_own_modality_output_is_still_reported(self, run_clinical) -> None:
        """The strip must not take the agent's own provenance with it."""
        out = run_clinical({"image_b64": "x", "modality": "image"})
        assert out["metadata"]["mode"] == "mri_image"
        assert out["metadata"]["vision_fallback"] is True


class TestEveryAttachmentTypeReachesACapability:
    """The node this file used to test could not receive media at all."""

    @pytest.mark.parametrize(
        "key,expected_mode",
        [
            ("image_b64", "mri_image"),
            ("report_b64", "pdf_report"),
            ("audio_b64", "audio"),
        ],
    )
    def test_the_attachment_selects_its_handler(
        self, run_clinical, monkeypatch: pytest.MonkeyPatch, key: str, expected_mode: str
    ) -> None:
        from mao.agents import clinical_agent

        monkeypatch.setattr(
            clinical_agent,
            "_extract_pdf_text",
            lambda metadata: "Patient was started on donepezil.",
        )
        monkeypatch.setattr(clinical_agent, "_summarize_report", lambda *a: "summary")
        monkeypatch.setattr(clinical_agent, "_extract_structured_fields", lambda t: {})

        out = run_clinical({key: "payload"})
        assert out["metadata"]["mode"] == expected_mode
