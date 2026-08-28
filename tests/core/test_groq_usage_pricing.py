"""Cost accounting must source prices from the ModelRegistry, not a duplicate table (P1-18)."""
from __future__ import annotations

import pytest

from mao.core import groq_usage
from mao.providers.gateway import registry


@pytest.fixture(autouse=True)
def _reset_tracker():
    groq_usage.tracker.reset()
    yield
    groq_usage.tracker.reset()


def test_no_duplicate_literal_price_table() -> None:
    """The literal `_PRICE_PER_1M` table duplicated registry metadata and went stale (P1-18)."""
    assert not hasattr(groq_usage, "_PRICE_PER_1M")


def test_price_matches_the_registry_record() -> None:
    record = registry().get("llama-3.1-8b-instant")
    assert record is not None
    price = groq_usage.price_for("llama-3.1-8b-instant")
    assert price.input_per_1m == pytest.approx(record.cost_per_1m_input_usd)
    assert price.output_per_1m == pytest.approx(record.cost_per_1m_output_usd)
    assert price.source == "registry"


def test_recorded_cost_uses_registry_pricing() -> None:
    record = registry().get("llama-3.3-70b-versatile")
    assert record is not None
    groq_usage.tracker.record("llama-3.3-70b-versatile", 1_000_000, 1_000_000)
    summary = groq_usage.tracker.get_summary()
    expected = record.cost_per_1m_input_usd + record.cost_per_1m_output_usd
    assert summary["estimated_cost_usd"] == pytest.approx(expected, rel=1e-6)


def test_unknown_model_degrades_gracefully() -> None:
    price = groq_usage.price_for("not-a-real-model-id")
    assert price.source == "unknown"
    assert price.input_per_1m > 0
    groq_usage.tracker.record("not-a-real-model-id", 1000, 1000)
    summary = groq_usage.tracker.get_summary()
    assert summary["estimated_cost_usd"] > 0
    assert "not-a-real-model-id" in summary["per_model"]


def test_retired_model_is_not_priced_from_a_stale_literal() -> None:
    """`llama-3.1-70b-versatile` is RETIRED and carries no registry price (P1-18)."""
    price = groq_usage.price_for("llama-3.1-70b-versatile")
    assert price.source != "registry"


def test_summary_shape_is_unchanged() -> None:
    """/usage documents per_model as {input_tokens, output_tokens, requests, cost_usd}."""
    groq_usage.tracker.record("llama-3.1-8b-instant", 10, 20)
    summary = groq_usage.tracker.get_summary()
    assert set(summary) == {
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "per_model",
    }
    assert set(summary["per_model"]["llama-3.1-8b-instant"]) == {
        "input_tokens",
        "output_tokens",
        "requests",
        "cost_usd",
    }
