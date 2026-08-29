"""arch-M4 / scope item 3 — every agent goes through the model gateway.

The gateway existed and was correct, but only three call sites used it. Seven
agents resolved a role to an id and then called `mao.core.llm` directly, so two
provider layers coexisted and the legacy one carried most traffic. Bypassing
`gateway.complete()` also means no `Completion`, which is why the trace's token
and cost fields were structurally dead (arch-M1).

arch-M1 — `prompt_ref` was written into the *nested* `state["verification_trace"]`
and read from the *top level*, so it was always empty.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import mao.agents
from mao.providers import gateway
from mao.providers.llm.base import ProviderResponse
from mao.providers.registry import ModelRole


def _agent_modules() -> list[str]:
    return [
        f"mao.agents.{m.name}"
        for m in pkgutil.iter_modules(mao.agents.__path__)
        if not m.name.startswith("_")
    ]


class TestNoAgentTalksToTheLegacyProviderLayer:
    @pytest.mark.parametrize("module_name", _agent_modules())
    def test_agent_does_not_import_core_llm(self, module_name: str) -> None:
        source = inspect.getsource(importlib.import_module(module_name))
        assert "mao.core.llm" not in source, (
            f"{module_name} bypasses the model gateway"
        )
        assert "mao.core import llm" not in source

    @pytest.mark.parametrize("module_name", _agent_modules())
    def test_agent_does_not_import_a_provider_sdk(self, module_name: str) -> None:
        source = inspect.getsource(importlib.import_module(module_name))
        assert "from groq import" not in source
        assert "import groq" not in source


class TestTheGatewayReportsRealUsage:
    def test_completion_carries_token_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Provider:
            name = "stub"

            def complete(self, **kwargs: object) -> ProviderResponse:
                return ProviderResponse(text="hi", input_tokens=11, output_tokens=7)

        gateway.set_provider(_Provider())
        try:
            done = gateway.complete(role=ModelRole.GENERAL_SYNTHESIS, messages=[])
        finally:
            gateway.reset_provider()

        assert (done.input_tokens, done.output_tokens) == (11, 7)

    def test_the_groq_provider_propagates_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The provider wrapper used to drop usage on the floor."""
        from mao.providers.llm import groq_provider

        monkeypatch.setattr(
            "mao.core.llm.chat_with_usage",
            lambda **kwargs: ("text", 21, 9, False),
        )
        out = groq_provider.GroqChatProvider().complete(
            model_id="m", messages=[], temperature=0.0, max_tokens=10
        )
        assert (out.input_tokens, out.output_tokens) == (21, 9)
        assert out.truncated is False

    def test_the_groq_provider_propagates_truncation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`finish_reason == "length"` must survive to the caller.

        On a reasoning model an under-budgeted call returns HTTP 200 with an
        empty string, which is indistinguishable from a model that chose to say
        nothing unless this flag arrives with it — the exact ambiguity that hid
        Wave 6 blocker 3.
        """
        from mao.providers.llm import groq_provider

        monkeypatch.setattr(
            "mao.core.llm.chat_with_usage",
            lambda **kwargs: ("", 21, 300, True),
        )
        out = groq_provider.GroqChatProvider().complete(
            model_id="m", messages=[], temperature=0.0, max_tokens=10
        )
        assert out.truncated is True
        assert out.text == ""


class TestUsageIsCollectedPerRequest:
    def test_completions_accumulate_into_the_active_collector(self) -> None:
        from mao.providers import usage

        class _Provider:
            name = "stub"

            def complete(self, **kwargs: object) -> ProviderResponse:
                return ProviderResponse(text="x", input_tokens=5, output_tokens=3)

        gateway.set_provider(_Provider())
        try:
            with usage.collecting() as totals:
                gateway.complete(role=ModelRole.GENERAL_SYNTHESIS, messages=[])
                gateway.complete(role=ModelRole.SAFETY_JUDGE, messages=[])
        finally:
            gateway.reset_provider()

        assert totals.input_tokens == 10
        assert totals.output_tokens == 6
        assert totals.estimated_cost_usd >= 0.0

    def test_a_completion_outside_a_collector_is_harmless(self) -> None:
        class _Provider:
            name = "stub"

            def complete(self, **kwargs: object) -> ProviderResponse:
                return ProviderResponse(text="x", input_tokens=5, output_tokens=3)

        gateway.set_provider(_Provider())
        try:
            gateway.complete(role=ModelRole.GENERAL_SYNTHESIS, messages=[])
        finally:
            gateway.reset_provider()


class TestTraceFieldsAreNoLongerStructurallyDead:
    def _trace(self, result: dict) -> object:
        from mao.api.tracing import build_trace

        return build_trace(trace_id="t1", result=result, latency_ms=12.0)

    def test_prompt_ref_is_read_from_where_it_is_written(self) -> None:
        """It was written nested and read top-level, so it was always empty."""
        trace = self._trace(
            {
                "intent": "graphrag",
                "verification_trace": {"prompt_ref": "judge.safety@1.0.0"},
            }
        )
        assert trace.prompt_ref == "judge.safety@1.0.0"

    def test_a_top_level_prompt_ref_still_wins_when_present(self) -> None:
        trace = self._trace({"intent": "graphrag", "prompt_ref": "router.classify@1.0.0"})
        assert trace.prompt_ref == "router.classify@1.0.0"

    def test_token_counts_reach_the_trace(self) -> None:
        trace = self._trace(
            {"intent": "graphrag", "llm_usage": {"input_tokens": 120, "output_tokens": 45}}
        )
        assert trace.input_tokens == 120
        assert trace.output_tokens == 45

    def test_estimated_cost_reaches_the_trace(self) -> None:
        trace = self._trace(
            {
                "intent": "graphrag",
                "llm_usage": {
                    "input_tokens": 1000,
                    "output_tokens": 1000,
                    "estimated_cost_usd": 0.0042,
                },
            }
        )
        assert trace.estimated_cost_usd == pytest.approx(0.0042)

    def test_a_trace_without_usage_is_still_valid(self) -> None:
        trace = self._trace({"intent": "graphrag"})
        assert trace.input_tokens == 0
        assert trace.estimated_cost_usd == 0.0
