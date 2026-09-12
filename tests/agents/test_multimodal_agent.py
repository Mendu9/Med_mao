"""Multimodal agent tests — P1-18: no literal or retired model ids."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tests.agents.gateway_stub import _completion

from mao.agents import multimodal_agent as mm
from mao.prompts import get_prompt
from mao.providers.gateway import model_id_for
from mao.providers.registry import ModelRole

RETIRED_VISION_ID = "llama-3.2-11b-vision-preview"


def test_module_source_contains_no_retired_model_id() -> None:
    src = Path(mm.__file__).read_text(encoding="utf-8")
    assert RETIRED_VISION_ID not in src


def test_module_holds_no_hardcoded_vision_model_constant() -> None:
    assert not hasattr(mm, "_VISION_MODEL")


def test_the_image_call_is_not_made_at_all() -> None:
    """Control decision M-2: raw patient-image EXTERNAL EGRESS is disabled.

    This asserted that the vision call resolved `ModelRole.VISION` from the
    registry rather than naming a retired id - which it did, correctly. What it
    could not see is that the call declared NO trust class and so took
    `complete()`'s default of SAFE_DERIVED_TEXT for a base64 patient scan
    (A-5, ADV16-7): "de-identified text minted by the protected input
    boundary", for a payload that had been through no boundary, had no
    InputChannel origin, and that `gateway._outgoing_text` cannot read because
    it collects only the text parts of a multipart turn.

    The role resolution it guarded is still asserted, below and unchanged, by
    the tests that read the module source - those do not need the call to
    happen. What is asserted here now is that no call happens.
    """
    with patch("mao.providers.gateway.complete") as chat:
        _, meta = mm.handle_image("what is this?", {"image_b64": "AAA"}, "")

    assert chat.call_count == 0
    assert meta["error"] == "image_disabled_phase_1"
    assert model_id_for(ModelRole.VISION) != RETIRED_VISION_ID


def test_lifting_the_gate_would_restore_the_vision_role_resolution() -> None:
    """Mutation control, and the record of what the re-enable gate inherits.

    Proves the test above passes because of `IMAGE_ENABLED` and not because the
    vision path is broken - and keeps the original role assertion alive against
    the day imagery is re-enabled under a truthful trust class.
    """
    with patch.object(mm, "IMAGE_ENABLED", True), patch(
        "mao.providers.gateway.complete", return_value=_completion("a brain MRI")
    ) as chat:
        answer, meta = mm.handle_image("what is this?", {"image_b64": "AAA"}, "")

    assert answer == "a brain MRI"
    assert chat.call_args.kwargs["role"] is ModelRole.VISION
    assert "model" not in chat.call_args.kwargs
    assert meta["vision_model"] == "stub-model"


def test_vision_system_prompt_comes_from_the_registry() -> None:
    """Unchanged in force, asserted behind the M-2 gate.

    The prompt must still come from the registry rather than an inline string,
    so that re-enabling imagery later does not also resurrect unversioned
    prompt text. `IMAGE_ENABLED` is lifted here only to reach the call.
    """
    with patch.object(mm, "IMAGE_ENABLED", True), patch(
        "mao.providers.gateway.complete", return_value=_completion("ok")
    ) as chat:
        mm.handle_image("what is this?", {"image_b64": "AAA"}, "")

    system = chat.call_args.kwargs["messages"][0]["content"]
    assert get_prompt("clinical.vision").template in system


def test_the_disabled_vision_call_would_declare_its_trust_class_explicitly() -> None:
    """The A-5 root cause, pinned so re-enabling cannot reintroduce it.

    A trust class a call site does not state is one the signature asserts on
    its behalf. Whatever class imagery is eventually carried under, the call
    site must name it.
    """
    with patch.object(mm, "IMAGE_ENABLED", True), patch(
        "mao.providers.gateway.complete", return_value=_completion("ok")
    ) as chat:
        mm.handle_image("what is this?", {"image_b64": "AAA"}, "")

    assert "trust_class" in chat.call_args.kwargs


def test_module_holds_no_inline_vision_system_prompt() -> None:
    assert not hasattr(mm, "_VISION_SYSTEM")
