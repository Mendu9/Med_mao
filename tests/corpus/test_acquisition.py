"""The acquisition policy must make corpus expansion reproducible.

Expansion from V1 to V2/V3/V4 must be a change of target counts against a fixed,
versioned query set — never a manual paper-selection exercise.
"""
from __future__ import annotations

import pytest

from mao.corpus.acquisition import load_policy, plan_expansion


class TestPolicyLoading:
    def test_loads_the_versioned_policy(self) -> None:
        policy = load_policy()
        assert policy.policy_version
        assert policy.categories

    def test_every_category_carries_queries(self) -> None:
        policy = load_policy()
        for name, queries in policy.categories.items():
            assert queries, f"category {name} has no queries"

    def test_policy_is_stable_across_loads(self) -> None:
        """A reproducible policy must not depend on dict or set ordering."""
        assert load_policy().query_fingerprint == load_policy().query_fingerprint

    def test_fingerprint_changes_when_queries_change(self) -> None:
        policy = load_policy()
        mutated = policy.with_categories({**policy.categories, "extra": ("new query",)})
        assert mutated.query_fingerprint != policy.query_fingerprint


class TestAdmissionFilters:
    def test_no_derivatives_licences_are_denied(self) -> None:
        policy = load_policy()
        assert not any("ND" in entry.split("-") for entry in policy.license_allow_list)

    def test_retracted_documents_are_excluded(self) -> None:
        assert load_policy().exclude_retracted is True


class TestScalingTiers:
    def test_defines_the_planned_growth_path(self) -> None:
        targets = [tier.target_documents for tier in load_policy().scaling_tiers]
        assert targets == [900, 3000, 5000, 10000]

    def test_tiers_increase_monotonically(self) -> None:
        targets = [tier.target_documents for tier in load_policy().scaling_tiers]
        assert targets == sorted(targets)


class TestExpansionPlanning:
    def test_plans_additional_documents_per_category(self) -> None:
        plan = plan_expansion(load_policy(), to_tier="V2", already_have=584)
        assert plan.target_documents == 3000
        assert plan.additional_documents == 3000 - 584

    def test_distributes_the_target_across_categories(self) -> None:
        plan = plan_expansion(load_policy(), to_tier="V2", already_have=0)
        assert sum(plan.per_category.values()) >= plan.target_documents - len(plan.per_category)

    def test_expansion_needs_no_manual_selection(self) -> None:
        """The plan is fully determined by the policy and the target tier."""
        first = plan_expansion(load_policy(), to_tier="V3", already_have=584)
        second = plan_expansion(load_policy(), to_tier="V3", already_have=584)
        assert first == second

    def test_already_satisfied_tier_requests_nothing(self) -> None:
        plan = plan_expansion(load_policy(), to_tier="V1", already_have=900)
        assert plan.additional_documents == 0

    def test_unknown_tier_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown tier"):
            plan_expansion(load_policy(), to_tier="V99", already_have=0)
