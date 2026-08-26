"""Contract tests for centralized safety policy ownership.

Safety thresholds must live in exactly one place and be versioned, so that
guardrails, council, and NLI gates cannot drift apart.
"""
from __future__ import annotations

import pytest

from mao.safety.policy import RiskLevel, SafetyPolicy, get_policy


class TestPolicyIdentity:
    def test_default_policy_is_versioned(self) -> None:
        policy = get_policy()
        assert policy.policy_version, "policy must carry a version for trace/provenance"

    def test_get_policy_returns_the_same_immutable_instance(self) -> None:
        assert get_policy() is get_policy()

    def test_policy_is_frozen(self) -> None:
        policy = get_policy()
        with pytest.raises(Exception):
            policy.nli_block_ratio = 0.99  # type: ignore[misc]


class TestThresholdsPreserveAuditedBehaviour:
    """The Phase 0 audit recorded these values; centralizing must not change them."""

    def test_nli_ratios_match_audited_values(self) -> None:
        policy = get_policy()
        assert policy.nli_warn_ratio == 0.30
        assert policy.nli_block_ratio == 0.70

    def test_judge_scores_match_audited_values(self) -> None:
        policy = get_policy()
        assert policy.judge_warn_score == 7
        assert policy.judge_block_score == 5

    def test_mri_confidence_gate_matches_audited_value(self) -> None:
        assert get_policy().mri_confidence_gate == 0.60


class TestRiskClassification:
    def test_clinical_intent_is_high_risk(self) -> None:
        assert get_policy().risk_for(intent="clinical") is RiskLevel.HIGH

    def test_chitchat_is_low_risk(self) -> None:
        assert get_policy().risk_for(intent="chitchat") is RiskLevel.LOW

    def test_attachments_escalate_risk_to_high(self) -> None:
        policy = get_policy()
        assert policy.risk_for(intent="graphrag", has_attachment=True) is RiskLevel.HIGH

    def test_unknown_intent_defaults_to_at_least_standard(self) -> None:
        assert get_policy().risk_for(intent="totally-unknown") is not RiskLevel.LOW


class TestVerificationRequirement:
    """Streaming must not be able to opt out of verification for risky traffic."""

    def test_high_risk_requires_verification(self) -> None:
        assert get_policy().requires_verification(RiskLevel.HIGH) is True

    def test_standard_risk_requires_verification(self) -> None:
        assert get_policy().requires_verification(RiskLevel.STANDARD) is True

    def test_low_risk_may_skip_verification(self) -> None:
        assert get_policy().requires_verification(RiskLevel.LOW) is False

    def test_disclaimer_is_required_for_high_risk(self) -> None:
        assert get_policy().requires_clinical_disclaimer(RiskLevel.HIGH) is True

    def test_disclaimer_not_required_for_low_risk(self) -> None:
        assert get_policy().requires_clinical_disclaimer(RiskLevel.LOW) is False


class TestPolicyOverride:
    def test_a_custom_policy_can_be_constructed_without_touching_the_default(self) -> None:
        strict = SafetyPolicy(policy_version="test-strict", nli_block_ratio=0.10)
        assert strict.nli_block_ratio == 0.10
        assert get_policy().nli_block_ratio == 0.70
