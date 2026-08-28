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


def test_image_call_uses_the_vision_role() -> None:
    with patch("mao.providers.gateway.complete", return_value=_completion("a brain MRI")) as chat:
        answer, meta = mm.handle_image("what is this?", {"image_b64": "AAA"}, "")

    assert answer == "a brain MRI"
    assert chat.call_args.kwargs["role"] is ModelRole.VISION
    assert "model" not in chat.call_args.kwargs
    # The id reported in metadata is whatever the gateway actually used.
    assert meta["vision_model"] == "stub-model"
    assert model_id_for(ModelRole.VISION) != RETIRED_VISION_ID


def test_vision_system_prompt_comes_from_the_registry() -> None:
    with patch("mao.providers.gateway.complete", return_value=_completion("ok")) as chat:
        mm.handle_image("what is this?", {"image_b64": "AAA"}, "")

    system = chat.call_args.kwargs["messages"][0]["content"]
    assert get_prompt("clinical.vision").template in system


def test_module_holds_no_inline_vision_system_prompt() -> None:
    assert not hasattr(mm, "_VISION_SYSTEM")
