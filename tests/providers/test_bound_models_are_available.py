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


def _router_budget() -> int:
    from mao.agents.router import _CLASSIFY_MAX_TOKENS

    return _CLASSIFY_MAX_TOKENS


def _domain_budget() -> int:
    from mao.agents.domain_classifier import _DOMAIN_MAX_TOKENS

    return _DOMAIN_MAX_TOKENS

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
    """Budget adequacy, asserted against values rather than against source text.

    These two checks were `inspect.getsource` string matches:

        assert "max_tokens=5" not in source
        assert "max_tokens=10" not in inspect.getsource(_classify)

    The architecture review called them out, and it was right: they pass at
    `max_tokens=6`, they never look at the two safety-critical budgets at all,
    and — the part that mattered — they cannot see which model is bound to the
    role. Adequacy is a relation between a budget and a model, so no assertion
    about the text of a call site can decide it. `openai/gpt-oss-safeguard-20b`
    spends up to 883 tokens reasoning before it answers; `qwen/qwen3.8-27b`
    spends 35. The same literal is fine for one and fatal for the other.

    What replaced them:
      - the relation, offline: tests/providers/test_reasoning_budget.py
      - the outcome, live:     tests/providers/test_live_call_sites.py
    """

    @pytest.mark.parametrize(
        "budget,minimum",
        [
            (_router_budget(), 16),
            (_domain_budget(), 16),
        ],
    )
    def test_a_one_word_label_has_room_for_a_short_preamble(
        self, budget: int, minimum: int
    ) -> None:
        assert budget >= minimum, (
            f"a {budget}-token answer budget leaves no room for a model that "
            "emits any preamble; an empty reply reads as 'never classified', "
            "which escalates every request to HIGH"
        )

    @pytest.mark.parametrize("role", [ModelRole.ROUTER_FAST, ModelRole.EXTRACTION_FAST])
    def test_the_analysis_channel_is_budgeted_separately(self, role: ModelRole) -> None:
        """The classifier budget must not have to cover the model's thinking."""
        assert registry().resolve(role).reasoning_overhead_tokens > 0


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
