"""L1 — multimodal echoed raw attachment payloads back into graph state.

`state["metadata"]` is persisted, traced, and logged. Spreading the inbound
metadata into it carried the raw base64 scan or voice sample along for the ride.
`clinical_agent` was fixed for this under P0-4; the same leak existed here.
"""
from __future__ import annotations

from unittest.mock import patch

from mao.safety.policy import ATTACHMENT_KEYS


def _run(metadata: dict) -> dict:
    from mao.agents.multimodal_agent import multimodal_node

    state = {
        "user_query": "what does this show?",
        "user_id": "u1",
        "metadata": metadata,
        "memory_context": "",
    }
    with patch("mao.agents.multimodal_agent.search_memories", return_value=""), \
         patch("mao.agents.multimodal_agent.save_memory"), \
         patch("mao.core.llm.chat", return_value="A description of the image."):
        return multimodal_node(state)


class TestNoRawPayloadInState:
    def test_image_payload_is_not_echoed(self) -> None:
        out = _run({"image_b64": "SENSITIVE_SCAN_BYTES", "modality": "image"})
        assert "SENSITIVE_SCAN_BYTES" not in str(out["metadata"])

    def test_audio_payload_is_not_echoed(self) -> None:
        out = _run({"audio_b64": "SENSITIVE_VOICE_BYTES", "modality": "audio"})
        assert "SENSITIVE_VOICE_BYTES" not in str(out["metadata"])

    def test_no_attachment_key_survives_into_state(self) -> None:
        out = _run({"image_b64": "x", "audio_b64": "y", "modality": "image"})
        for key in ATTACHMENT_KEYS:
            assert key not in out["metadata"], f"{key} leaked into graph state"

    def test_non_attachment_metadata_is_preserved(self) -> None:
        out = _run({"image_b64": "x", "modality": "image", "case_id": "abc-123"})
        assert out["metadata"].get("case_id") == "abc-123"

    def test_modality_is_still_reported(self) -> None:
        out = _run({"image_b64": "x", "modality": "image"})
        assert out["metadata"].get("modality") == "image"
