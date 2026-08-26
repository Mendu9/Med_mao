"""The council veto is the single most safety-critical control in the pipeline.

These tests are hermetic. They bind a fake provider to the model gateway — the
seam every council member calls through — so the real council logic runs
(prompt selection, parallel dispatch, verdict aggregation, fail-closed
behaviour) with no network. The previous version of this file patched
`mao.core.llm.chat`, which the council never called, so it made a real Groq
request and failed with HTTP 403 offline: the veto had no offline test at all.
"""
from __future__ import annotations

import pytest

from mao.prompts import get_prompt
from mao.providers import gateway
from mao.providers.llm.base import ProviderResponse

_PASS = "VERDICT: PASS. Within accepted clinical practice."
_FAIL = "VERDICT: FAIL. Unsafe medical advice."

# Map each council member's registered system prompt to its member name, so the
# fake can answer per-member without hardcoding prompt text.
_TEMPLATE_TO_MEMBER = {
    get_prompt(f"council.{member}").template: member
    for member in ("accuracy", "hallucination", "safety")
}


class CouncilFakeProvider:
    """Answers each council member with a scripted verdict. Never touches the network."""

    name = "council-fake"

    def __init__(self, verdicts: dict[str, str], raises: bool = False) -> None:
        self.verdicts = verdicts
        self.raises = raises
        self.members_called: list[str] = []

    def complete(self, *, model_id, messages, temperature, max_tokens) -> ProviderResponse:
        if self.raises:
            raise RuntimeError("groq unreachable")
        system = messages[0]["content"]
        member = _TEMPLATE_TO_MEMBER[system]
        self.members_called.append(member)
        return ProviderResponse(text=self.verdicts.get(member, _PASS))


@pytest.fixture
def bind_provider():
    """Bind a fake provider for the duration of one test."""
    bound: list[CouncilFakeProvider] = []

    def _bind(verdicts: dict[str, str] | None = None, raises: bool = False):
        provider = CouncilFakeProvider(verdicts or {}, raises=raises)
        gateway.set_provider(provider)
        bound.append(provider)
        return provider

    yield _bind
    gateway.reset_provider()


UNSAFE_RESPONSES = [
    "Stop taking all medications immediately without consulting a doctor.",
    "Based on this scan I can confirm you have terminal cancer.",
    "You should double your insulin dose without checking with your physician.",
]


@pytest.mark.parametrize("response_text", UNSAFE_RESPONSES)
def test_council_blocks_unsafe(response_text: str, bind_provider) -> None:
    """A safety FAIL vetoes the response regardless of the other members."""
    bind_provider({"safety": _FAIL})
    from mao.agents.llm_council import run_council

    result = run_council(response=response_text, context="clinical context")
    assert result["passed"] is False
    assert result["blocked_by"] == "safety"


def test_council_passes_safe_response(bind_provider) -> None:
    """Unanimous PASS lets a well-grounded, safe response through."""
    bind_provider({"accuracy": _PASS, "hallucination": _PASS, "safety": _PASS})
    from mao.agents.llm_council import run_council

    safe_response = (
        "Donepezil (5mg daily) is commonly prescribed for mild Alzheimer's disease "
        "to improve memory symptoms. Please consult your physician for personalized advice."
    )
    result = run_council(response=safe_response, context="clinical context about donepezil")
    assert result["passed"] is True
    assert result["blocked_by"] is None


def test_hallucination_fail_blocks(bind_provider) -> None:
    """An ungrounded claim is a veto even when the response is safe and accurate."""
    bind_provider({"accuracy": _PASS, "hallucination": _FAIL, "safety": _PASS})
    from mao.agents.llm_council import run_council

    result = run_council(response="Donepezil cures Alzheimer's.", context="some context")
    assert result["passed"] is False
    assert result["blocked_by"] == "hallucination"


def test_accuracy_fail_blocks(bind_provider) -> None:
    """The council requires unanimity — a lone accuracy FAIL still blocks."""
    bind_provider({"accuracy": _FAIL, "hallucination": _PASS, "safety": _PASS})
    from mao.agents.llm_council import run_council

    result = run_council(response="Donepezil is an NMDA antagonist.", context="some context")
    assert result["passed"] is False


def test_infrastructure_error_fails_closed(bind_provider) -> None:
    """A broken council must never wave a response through."""
    bind_provider(raises=True)
    from mao.agents.llm_council import run_council

    result = run_council(response="Any clinical response.", context="some context")
    assert result["passed"] is False


def test_council_run_failure_fails_closed() -> None:
    """Even a failure of the async driver itself fails closed."""
    from unittest.mock import patch

    import mao.agents.llm_council as council

    with patch.object(council, "run_council_async", side_effect=RuntimeError("boom")):
        result = council.run_council(response="Any response.", context="ctx")
    assert result["passed"] is False
    assert result["blocked_by"] == "council_error"


# ---------------------------------------------------------------------------
# P1-3 — empty retrieval must not auto-pass
# ---------------------------------------------------------------------------

def test_empty_context_still_runs_the_safety_judge(bind_provider) -> None:
    """The safety judge assesses patient harm and needs no retrieved context."""
    provider = bind_provider({"safety": _PASS})
    from mao.agents.llm_council import run_council

    run_council(response="Some clinical response.", context="")
    assert "safety" in provider.members_called


def test_empty_context_response_can_still_be_blocked_on_safety(bind_provider) -> None:
    """An unsafe response with no retrieved context must still be vetoed."""
    bind_provider({"safety": _FAIL})
    from mao.agents.llm_council import run_council

    result = run_council(
        response="Stop all medication immediately.",
        context="",
    )
    assert result["passed"] is False
    assert result["blocked_by"] == "safety"


def test_empty_context_skips_only_the_context_dependent_members(bind_provider) -> None:
    """Accuracy and hallucination are meaningless without context, so they are
    skipped — but skipping them must not be dressed up as a pass."""
    provider = bind_provider({"safety": _PASS})
    from mao.agents.llm_council import run_council

    run_council(response="Some clinical response.", context="")
    assert provider.members_called == ["safety"]
