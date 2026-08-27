"""Intent vocabulary and state contract tests.

Covers P0-3 (no ``sql`` intent), P2-3 (no dead ``code`` intent) and the new
``risk_level`` field.
"""
from __future__ import annotations

from typing import get_type_hints

from mao.core import state as state_mod
from mao.core.state import ALL_INTENTS, MAOState, make_initial_state, risk_level_of
from mao.safety.policy import RiskLevel


def test_sql_intent_is_gone() -> None:
    assert "sql" not in ALL_INTENTS
    assert not hasattr(state_mod, "INTENT_SQL")


def test_code_intent_is_gone() -> None:
    assert "code" not in ALL_INTENTS
    assert not hasattr(state_mod, "INTENT_CODE")


def test_remaining_intents() -> None:
    assert set(ALL_INTENTS) == {
        "summarize",
        "graphrag",
        "tool",
        "multimodal",
        "critic",
        "clinical",
        "fallback",
        "chitchat",
    }


def test_state_declares_risk_level() -> None:
    assert get_type_hints(MAOState)["risk_level"] is str


def test_initial_state_has_risk_level_key() -> None:
    assert "risk_level" in make_initial_state("hi", "u1")


# ---------------------------------------------------------------------------
# risk_level_of — fail-safe coercion
# ---------------------------------------------------------------------------

def test_risk_level_of_reads_the_written_value() -> None:
    assert risk_level_of({"risk_level": "low"}) is RiskLevel.LOW
    assert risk_level_of({"risk_level": "high"}) is RiskLevel.HIGH


def test_risk_level_of_defaults_to_standard_when_absent() -> None:
    """Absent risk must never be read as LOW — LOW is the only level that
    may skip verification."""
    assert risk_level_of({}) is RiskLevel.STANDARD


def test_risk_level_of_defaults_to_standard_when_unrecognised() -> None:
    assert risk_level_of({"risk_level": "banana"}) is RiskLevel.STANDARD
