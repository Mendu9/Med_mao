"""A call site's budget is a budget for the ANSWER, not for the model's thinking.

Wave 6 blocker 3. `SAFETY_JUDGE` is bound to `openai/gpt-oss-safeguard-20b`, a
reasoning model that emits a long analysis channel before its answer. Every
safety call site kept a budget sized for the previous non-reasoning model, so
the whole budget went on the analysis channel and `message.content` came back
empty. The fail-closed parsers then blocked SAFE answers: 5/5 ordinary clinical
questions were refused.

Measured live at the frozen SHA, real prompts at their real call-site budgets:

    judge.safety                    @300 -> finish=length out=300 content=0 chars
    council.accuracy                @200 -> finish=length out=200 content=0 chars
    council.hallucination           @200 -> finish=length out=200 content=0 chars
    domain_supervisor.reconcile     @500 -> finish=length out=500 content=0 chars

The naive repair — raise the three constants the reviews named — is the same
mistake one layer up. There are 18 call sites, the provider's entire catalogue
is reasoning models, and the overhead is a property of the *bound model*, not of
the call site. Measured worst-case analysis channel, real prompts, 8192 budget:

    qwen/qwen3.8-27b                 35 tokens
    openai/gpt-oss-120b             437 tokens
    openai/gpt-oss-safeguard-20b    883 tokens

which is also why `ROUTER_FAST @64` worked and looked like evidence that only
the judge was mis-budgeted: the router is bound to qwen, which barely reasons.

So the overhead is declared on the model record and added by the gateway. A call
site asks for the answer it needs; rebinding a role re-sizes every one of its
call sites automatically, which is precisely the Wave 5 -> Wave 6 regression
(fail-closed parsing + rebinding to a live model, composed without re-measuring).
"""
from __future__ import annotations

import pytest

from mao.providers import gateway
from mao.providers.llm.base import ProviderResponse
from mao.providers.registry import ModelRole, ModelStatus


class _RecordingProvider:
    name = "recording"

    def __init__(self, text: str = "ok", finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.max_tokens_seen: list[int] = []

    def complete(self, *, model_id, messages, temperature, max_tokens) -> ProviderResponse:
        self.max_tokens_seen.append(max_tokens)
        return ProviderResponse(
            text=self.text,
            input_tokens=1,
            output_tokens=1,
            truncated=self.finish_reason == "length",
        )


@pytest.fixture
def recording():
    provider = _RecordingProvider()
    gateway.set_provider(provider)
    yield provider
    gateway.reset_provider()


# ---------------------------------------------------------------------------
# The catalogue declares the overhead
# ---------------------------------------------------------------------------

class TestEveryActiveModelDeclaresItsReasoningOverhead:
    @pytest.mark.parametrize("role", list(ModelRole))
    def test_the_bound_record_declares_an_overhead(self, role: ModelRole) -> None:
        record = gateway.resolve(role)
        assert record.reasoning_overhead_tokens >= 0

    def test_the_safety_judge_model_declares_a_measured_overhead(self) -> None:
        """883 tokens were observed live; the declared value must cover it."""
        record = gateway.resolve(ModelRole.SAFETY_JUDGE)
        assert record.reasoning_overhead_tokens >= 883, (
            "the declared overhead is below the worst overhead measured live, "
            "so the safety call sites will truncate again"
        )

    def test_an_unknown_operator_supplied_id_is_treated_as_a_reasoning_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under-budgeting fails silently; over-budgeting only raises a ceiling."""
        from mao.providers.registry import ModelRegistry

        monkeypatch.setenv(ModelRole.SAFETY_JUDGE.env_var, "some/unknown-model")
        registry = ModelRegistry.default()
        record = registry.resolve(ModelRole.SAFETY_JUDGE)

        assert record.model_id == "some/unknown-model"
        assert record.status is ModelStatus.ACTIVE
        assert record.reasoning_overhead_tokens >= 883


# ---------------------------------------------------------------------------
# The gateway applies it
# ---------------------------------------------------------------------------

class TestTheGatewayBudgetsForTheBoundModel:
    def test_the_provider_is_asked_for_the_answer_plus_the_overhead(
        self, recording
    ) -> None:
        overhead = gateway.resolve(ModelRole.SAFETY_JUDGE).reasoning_overhead_tokens
        gateway.complete(
            role=ModelRole.SAFETY_JUDGE,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=200,
        )
        assert recording.max_tokens_seen == [200 + overhead]

    def test_the_answer_budget_is_what_the_call_site_asked_for(self, recording) -> None:
        """The overhead is added, never substituted — a caller that asks for 1024
        tokens of clinical synthesis must still get room for 1024."""
        overhead = gateway.resolve(ModelRole.CLINICAL_SYNTHESIS).reasoning_overhead_tokens
        gateway.complete(
            role=ModelRole.CLINICAL_SYNTHESIS,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
        )
        assert recording.max_tokens_seen[0] - overhead == 1024

    def test_a_low_overhead_model_is_not_penalised(self, recording) -> None:
        """qwen barely reasons; the router must not start paying judge-sized
        budgets just because they share a gateway."""
        judge = gateway.resolve(ModelRole.SAFETY_JUDGE).reasoning_overhead_tokens
        router = gateway.resolve(ModelRole.ROUTER_FAST).reasoning_overhead_tokens
        assert router < judge


# ---------------------------------------------------------------------------
# Truncation is a failure, not an empty answer
# ---------------------------------------------------------------------------

class TestTruncationIsDistinguishableFromSilence:
    def test_completion_reports_truncation(self) -> None:
        provider = _RecordingProvider(text="", finish_reason="length")
        gateway.set_provider(provider)
        try:
            completion = gateway.complete(
                role=ModelRole.SAFETY_JUDGE,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
            assert completion.truncated is True
        finally:
            gateway.reset_provider()

    def test_a_complete_answer_is_not_marked_truncated(self, recording) -> None:
        completion = gateway.complete(
            role=ModelRole.SAFETY_JUDGE,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
        assert completion.truncated is False


class TestATruncatedEmptyReplyIsRetriedNotAccepted:
    """The declared overhead is sized to measurement, so the tail needs a retry.

    Sizing it to worst-case-plus-margin instead would be paid on every call:
    the provider's rate limiter charges the requested ceiling, not the tokens
    produced — a 429 during remediation read "Limit 200000, Used 199385,
    Requested 2333". Retrying the rare cut-off is cheaper than pre-paying for
    it on every request, and both avoid the silent empty string.
    """

    def test_an_empty_truncated_reply_is_retried_with_more_room(self) -> None:
        provider = _RecordingProvider(text="", finish_reason="length")
        gateway.set_provider(provider)
        try:
            gateway.complete(
                role=ModelRole.SAFETY_JUDGE,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=300,
            )
        finally:
            gateway.reset_provider()

        assert len(provider.max_tokens_seen) == 2, "the truncated call was accepted"
        assert provider.max_tokens_seen[1] > provider.max_tokens_seen[0]

    def test_a_truncated_reply_that_said_something_is_not_retried(self) -> None:
        """A cut-off answer is still an answer; re-asking costs quota for nothing."""
        provider = _RecordingProvider(text='{"safety": 8}', finish_reason="length")
        gateway.set_provider(provider)
        try:
            gateway.complete(
                role=ModelRole.SAFETY_JUDGE,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=300,
            )
        finally:
            gateway.reset_provider()

        assert len(provider.max_tokens_seen) == 1

    def test_a_normal_reply_is_not_retried(self, recording) -> None:
        gateway.complete(
            role=ModelRole.SAFETY_JUDGE,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=300,
        )
        assert len(recording.max_tokens_seen) == 1

    def test_the_retry_does_not_loop(self) -> None:
        """Exactly one retry. A model that cannot answer must not be asked
        forever on the request path."""
        provider = _RecordingProvider(text="", finish_reason="length")
        gateway.set_provider(provider)
        try:
            completion = gateway.complete(
                role=ModelRole.SAFETY_JUDGE,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=300,
            )
        finally:
            gateway.reset_provider()

        assert len(provider.max_tokens_seen) == 2
        assert completion.text == ""
        assert completion.truncated is True
