"""The gateway is the only call surface business logic may use.

Agents ask for a role. They never name a model id and never import a provider
SDK. These tests use a fake provider, so they never touch the network.
"""
from __future__ import annotations

import pytest

from mao.providers import gateway
from mao.providers.llm.base import ChatProvider, ProviderResponse
from mao.providers.registry import ModelRole


class FakeProvider:
    """Records what the gateway asked for."""

    name = "fake"

    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.calls: list[dict] = []

    def complete(self, *, model_id, messages, temperature, max_tokens) -> ProviderResponse:
        self.calls.append(
            {
                "model_id": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return ProviderResponse(text=self.text, input_tokens=11, output_tokens=7)


@pytest.fixture
def fake() -> FakeProvider:
    provider = FakeProvider()
    gateway.set_provider(provider)
    yield provider
    gateway.reset_provider()


class TestRoleAddressedCompletion:
    def test_fake_provider_satisfies_the_protocol(self, fake: FakeProvider) -> None:
        assert isinstance(fake, ChatProvider)

    def test_gateway_resolves_the_role_to_a_model_id(self, fake: FakeProvider) -> None:
        gateway.complete(role=ModelRole.SAFETY_JUDGE, messages=[{"role": "user", "content": "hi"}])
        assert fake.calls[0]["model_id"] == gateway.model_id_for(ModelRole.SAFETY_JUDGE)

    def test_completion_carries_provenance_for_the_trace(self, fake: FakeProvider) -> None:
        result = gateway.complete(
            role=ModelRole.GENERAL_SYNTHESIS, messages=[{"role": "user", "content": "hi"}]
        )
        assert result.text == "ok"
        assert result.role is ModelRole.GENERAL_SYNTHESIS
        assert result.provider == "fake"
        assert result.input_tokens == 11
        assert result.output_tokens == 7

    def test_cost_is_derived_from_the_registry_record(self, fake: FakeProvider) -> None:
        result = gateway.complete(
            role=ModelRole.CLINICAL_SYNTHESIS, messages=[{"role": "user", "content": "hi"}]
        )
        record = gateway.resolve(ModelRole.CLINICAL_SYNTHESIS)
        expected = (
            11 * record.cost_per_1m_input_usd + 7 * record.cost_per_1m_output_usd
        ) / 1_000_000
        assert result.estimated_cost_usd == pytest.approx(expected)

    def test_caller_may_not_pass_a_raw_model_id(self) -> None:
        """Business logic must not be able to bypass role resolution."""
        import inspect

        assert "model_id" not in inspect.signature(gateway.complete).parameters


class TestRetiredModelsNeverReachTheProvider:
    def test_a_retired_binding_is_substituted_before_the_call(self, fake: FakeProvider) -> None:
        from mao.providers.registry import ModelRegistry, ModelRecord, Modality, ModelStatus

        gateway._registry = ModelRegistry(  # noqa: SLF001 - test seam
            records={
                "dead": ModelRecord(
                    provider="fake",
                    model_id="dead",
                    modality=Modality.TEXT,
                    context_limit=1,
                    status=ModelStatus.RETIRED,
                    fallbacks=("alive",),
                ),
                "alive": ModelRecord(
                    provider="fake", model_id="alive", modality=Modality.TEXT, context_limit=1
                ),
            },
            role_bindings={ModelRole.VISION: "dead"},
        )
        try:
            gateway.complete(role=ModelRole.VISION, messages=[{"role": "user", "content": "x"}])
            assert fake.calls[0]["model_id"] == "alive"
        finally:
            gateway.reset_registry()
