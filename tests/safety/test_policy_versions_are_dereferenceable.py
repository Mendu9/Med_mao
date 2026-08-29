"""Scope item 23 — PolicyRegistry, and scope item 6's capability lookup.

The architecture review recorded both as PARTIAL: policy *version metadata* was
done and threaded into the cache key and every trace, but "no PolicyRegistry
exists (one `SafetyPolicy` + `get_policy()`)"; and the ToolRegistry existed with
"no CapabilityRegistry".

What was actually missing in each case was a lookup, not a container:

  - every trace records `policy_version`, and nothing could resolve that string
    back to the thresholds it named. A trace store whose policy references
    cannot be dereferenced is not a learning substrate, and the learning data
    plane is half of this phase.

  - every ToolSpec declares a `capability`, and nothing could query by it, so
    "which tool provides search" had no answer despite the data being present.

A CapabilityRegistry is deliberately not built as a second registry over the
same ToolSpecs — that adds a synchronisation problem and no answer. The lookup
is on the registry that already owns the declarations.
"""
from __future__ import annotations

import pytest

from mao.safety.policy import POLICY_VERSION, PolicyRegistry, SafetyPolicy, get_policy, registry
from mao.tools import registry as tool_registry


class TestARecordedPolicyVersionCanBeResolved:
    def test_the_active_version_is_registered(self) -> None:
        assert POLICY_VERSION in registry().versions()

    def test_a_version_resolves_to_the_policy_it_names(self) -> None:
        resolved = registry().get(POLICY_VERSION)
        assert resolved.judge_block_score == get_policy().judge_block_score
        assert resolved.nli_block_ratio == get_policy().nli_block_ratio

    def test_the_version_a_trace_records_is_the_one_that_resolves(self) -> None:
        """Binds the trace field to the registry, not to a coincidence."""
        from mao.api.tracing import build_trace

        trace = build_trace(trace_id="t", result={"intent": "graphrag"}, latency_ms=1.0)
        assert registry().get(trace.policy_version) is get_policy()

    def test_an_unknown_version_is_refused_not_guessed(self) -> None:
        with pytest.raises(KeyError):
            registry().get("1999.01-0")

    def test_get_policy_returns_the_active_policy(self) -> None:
        assert get_policy() is registry().active()


class TestAVersionIsImmutableOnceRegistered:
    """Rewriting a version silently changes what every trace citing it means."""

    def test_re_registering_different_values_under_one_version_is_refused(self) -> None:
        local = PolicyRegistry.default()
        with pytest.raises(ValueError):
            local.register(SafetyPolicy(judge_block_score=1))

    def test_re_registering_identical_values_is_harmless(self) -> None:
        local = PolicyRegistry.default()
        local.register(SafetyPolicy())
        assert local.versions() == [POLICY_VERSION]

    def test_a_new_version_is_recorded_alongside_the_old(self) -> None:
        local = PolicyRegistry.default()
        local.register(SafetyPolicy(policy_version="2026.09-1", judge_block_score=6))
        assert local.get("2026.09-1").judge_block_score == 6
        assert local.get(POLICY_VERSION).judge_block_score == 5

    def test_an_active_version_must_exist(self) -> None:
        with pytest.raises(ValueError):
            PolicyRegistry({}, active="nope")


class TestToolsAreAddressableByCapability:
    def test_capabilities_are_enumerable(self) -> None:
        assert tool_registry().capabilities()

    def test_every_registered_tool_is_reachable_by_its_capability(self) -> None:
        for spec in tool_registry().all():
            found = tool_registry().by_capability(spec.capability)
            assert spec.tool_id in {s.tool_id for s in found}

    def test_an_unknown_capability_returns_nothing_rather_than_raising(self) -> None:
        assert tool_registry().by_capability("telepathy") == []

    def test_capability_results_are_cheapest_first(self) -> None:
        for capability in tool_registry().capabilities():
            found = tool_registry().by_capability(capability)
            costs = [(s.cost_per_call_usd, s.typical_latency_ms) for s in found]
            assert costs == sorted(costs)

    def test_domain_scoping_works(self) -> None:
        alzheimer_tools = tool_registry().for_domain("alzheimer")
        assert {s.tool_id for s in alzheimer_tools} <= set(tool_registry().names())
        assert all("alzheimer" in s.domains for s in alzheimer_tools)
