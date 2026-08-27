"""There must be exactly one way to read risk out of graph state.

Track A and Track B each grew their own resolver while working in parallel.
Two resolvers is two chances to disagree about whether a request may skip
verification, so resolution lives in the policy module and both delegate to it.
"""
from __future__ import annotations

import pytest

from mao.safety.policy import RiskLevel, resolve_risk


class TestValidValues:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("low", RiskLevel.LOW),
            ("standard", RiskLevel.STANDARD),
            ("high", RiskLevel.HIGH),
        ],
    )
    def test_string_values_resolve(self, raw: str, expected: RiskLevel) -> None:
        assert resolve_risk({"risk_level": raw}) is expected

    def test_an_enum_value_passes_through(self) -> None:
        assert resolve_risk({"risk_level": RiskLevel.HIGH}) is RiskLevel.HIGH


class TestFailsSafe:
    """LOW is the only level allowed to skip verification, so it must never be
    reachable by accident."""

    @pytest.mark.parametrize(
        "raw",
        ["", None, "LOW", "unknown", "0", 0, 1, [], {}, object()],
    )
    def test_anything_unrecognised_resolves_to_standard(self, raw) -> None:
        assert resolve_risk({"risk_level": raw}) is RiskLevel.STANDARD

    def test_missing_key_resolves_to_standard(self) -> None:
        assert resolve_risk({}) is RiskLevel.STANDARD

    def test_a_non_dict_state_resolves_to_standard(self) -> None:
        assert resolve_risk(None) is RiskLevel.STANDARD

    def test_resolved_default_requires_verification(self) -> None:
        """The fail-safe default must actually be a verifying level."""
        from mao.safety.policy import get_policy

        assert get_policy().requires_verification(resolve_risk({})) is True

    def test_uppercase_is_not_silently_accepted_as_low(self) -> None:
        assert resolve_risk({"risk_level": "LOW"}) is not RiskLevel.LOW
