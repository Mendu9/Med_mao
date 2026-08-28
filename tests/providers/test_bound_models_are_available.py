"""Scope item 24 — every bound model id must be one the provider still serves.

Found by execution during remediation, and missed by both exit-gate reviews
because both audited code rather than calling the provider: *every* actively
bound model id returned HTTP 404 from Groq.

    llama-3.1-8b-instant                       404 does not exist
    llama-3.3-70b-versatile                    404 does not exist
    meta-llama/llama-4-scout-17b-16e-instruct  404 does not exist

The registry was built to record *retired* ids so they can be refused, which is
the right mechanism — but the ids it bound as ACTIVE were themselves no longer
served, so not one LLM call in the system could succeed. Phase 1's exit criterion
is "no known P0 correctness issue may remain", and a system that cannot reach a
model is not a working system.

The offline tests here run everywhere. The live check is marked `slow` and is
deselected in the default run, like the other network tests.
"""
from __future__ import annotations

import pytest

from mao.providers.gateway import registry
from mao.providers.registry import ModelRole, ModelStatus

# Ids proven dead by live probe on 2026-08-28. None may be bound to a role.
DECOMMISSIONED = {
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.2-11b-vision-preview",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
}


class TestNoRoleResolvesToADecommissionedModel:
    @pytest.mark.parametrize("role", list(ModelRole))
    def test_role_does_not_resolve_to_a_dead_id(self, role: ModelRole) -> None:
        assert registry().model_id_for(role) not in DECOMMISSIONED

    @pytest.mark.parametrize("model_id", sorted(DECOMMISSIONED))
    def test_dead_ids_are_recorded_as_retired(self, model_id: str) -> None:
        """Recorded, so an operator override to one of them is refused."""
        record = registry().get(model_id)
        assert record is not None, f"{model_id} is not in the catalogue"
        assert record.status is ModelStatus.RETIRED

    @pytest.mark.parametrize("role", list(ModelRole))
    def test_every_role_resolves_to_something_active(self, role: ModelRole) -> None:
        assert registry().resolve(role).status is ModelStatus.ACTIVE


class TestFastRolesToleratePreambleFreeBudgets:
    """The classifier asks for 5 tokens and the router for 10.

    A reasoning model spends that budget thinking and returns an empty string,
    so a tight budget silently turns into "the router never classified anything".
    Both the model choice and the budget have to be safe.
    """

    def test_domain_classifier_budget_is_not_hair_trigger(self) -> None:
        import inspect

        from mao.agents import domain_classifier

        source = inspect.getsource(domain_classifier._llm_classify)
        assert "max_tokens=5" not in source

    def test_router_budget_is_not_hair_trigger(self) -> None:
        import inspect

        from mao.agents.router import _classify

        assert "max_tokens=10" not in inspect.getsource(_classify)


@pytest.mark.slow
class TestLiveProviderAvailability:
    """Calls the provider. Deselected by default, like the other network tests."""

    @pytest.mark.parametrize("role", list(ModelRole))
    def test_every_bound_model_answers(self, role: ModelRole) -> None:
        from mao.providers import gateway

        completion = gateway.complete(
            role=role,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0,
            max_tokens=200,
        )
        assert completion.text.strip(), f"{role.name} -> {completion.model_id} returned nothing"
