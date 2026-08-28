"""Findings H2 (judge coercion) and H5 (duplicate risk resolver).

H2 — `_score` coerced any non-numeric judge value to 10, the maximally *safe*
score. A judge that correctly identified harm but wrote "unsafe" instead of a
number was recorded as perfectly safe, and the guardrail block branch could
never fire on it.

H5 — `mao/safety/verification.py` carried a second `resolve_risk` with lenient
semantics ("Low" -> LOW), and `output_guardrails` imported *that* one while
`mao/api/streaming.py` used the strict policy resolver. LOW is the only level
permitted to skip verification and the clinical disclaimer, so the two halves of
the P0-1 guarantee disagreed about what LOW means.
"""
from __future__ import annotations

import pytest

from mao.guardrails import output_guardrails as og
from mao.safety import policy as policy_mod
from mao.safety import verification as ver
from mao.safety.policy import RiskLevel


# ---------------------------------------------------------------------------
# H2 — a judge that answered must not be scored as safe when unparseable
# ---------------------------------------------------------------------------

class TestJudgeContentFailsClosed:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"safety": "unsafe", "groundedness": 1}',
            '{"safety": "low", "groundedness": 8}',
            '{"safety": null, "groundedness": 8}',
            '{"groundedness": 8}',
            '{"safety": "N/A"}',
        ],
    )
    def test_non_numeric_safety_scores_zero(self, raw: str) -> None:
        assert ver._parse_judge(raw)["safety"] == 0

    def test_a_zero_safety_score_blocks_in_the_guardrails(self) -> None:
        """The whole point: the block branch must actually be reachable."""
        from mao.safety.policy import get_policy

        assert ver._parse_judge('{"safety": "unsafe"}')["safety"] < (
            get_policy().judge_block_score
        )

    def test_unparseable_judge_output_scores_zero(self) -> None:
        assert ver._parse_judge("I think this is fine, honestly.")["safety"] == 0

    def test_empty_judge_output_scores_zero(self) -> None:
        assert ver._parse_judge("")["safety"] == 0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"safety": 9, "groundedness": 8}', 9),
            ('{"safety": 0, "groundedness": 0}', 0),
            ('{"safety": "7", "groundedness": 7}', 7),
            ('{"safety": 7.4, "groundedness": 7}', 7),
            ('{"safety": 99, "groundedness": 7}', 10),
            ('{"safety": -5, "groundedness": 7}', 0),
        ],
    )
    def test_valid_numeric_scores_survive(self, raw: str, expected: int) -> None:
        assert ver._parse_judge(raw)["safety"] == expected

    def test_provider_outage_still_fails_open_with_a_note(self) -> None:
        """Deliberate and documented: the council is the fail-closed backstop.

        This is only defensible because the council now requires an affirmative
        PASS (C2). A crashed provider must not blanket-block clinical traffic.
        """
        def _boom(**kwargs: object) -> object:
            raise RuntimeError("provider down")

        original = ver.gateway.complete
        ver.gateway.complete = _boom  # type: ignore[assignment]
        try:
            scores = ver._run_judge("some response", "premise")
        finally:
            ver.gateway.complete = original  # type: ignore[assignment]

        assert scores["safety"] == 10
        assert "judge unavailable" in scores["notes"]


# ---------------------------------------------------------------------------
# H5 — exactly one risk resolver
# ---------------------------------------------------------------------------

class TestOneRiskResolver:
    def test_verification_does_not_define_its_own_resolver(self) -> None:
        import inspect

        source = inspect.getsource(ver)
        assert "def resolve_risk" not in source

    def test_verification_uses_the_policy_resolver(self) -> None:
        assert ver.resolve_risk is policy_mod.resolve_risk

    def test_output_guardrails_uses_the_policy_resolver(self) -> None:
        assert og.resolve_risk is policy_mod.resolve_risk

    @pytest.mark.parametrize("raw", ["Low", " LOW ", "lo w", "bogus", None, 0, []])
    def test_only_an_exact_low_is_low(self, raw: object) -> None:
        assert policy_mod.resolve_risk({"risk_level": raw}) is not RiskLevel.LOW

    def test_an_exact_low_is_low(self) -> None:
        assert policy_mod.resolve_risk({"risk_level": "low"}) is RiskLevel.LOW

    def test_verification_does_not_relist_attachment_keys(self) -> None:
        import inspect

        assert '"image_b64"' not in inspect.getsource(ver)


class TestAbsentRiskEscalatesRatherThanAssumingAFloor:
    """Removing the duplicate resolver must not drop the intent fallback.

    Defaulting a clinical request with no `risk_level` to STANDARD silently
    drops its mandatory disclaimer. The resolver classifies instead.
    """

    def test_clinical_intent_without_a_risk_level_is_high(self) -> None:
        assert policy_mod.resolve_risk({"intent": "clinical"}) is RiskLevel.HIGH

    def test_an_attachment_without_a_risk_level_is_high(self) -> None:
        assert policy_mod.resolve_risk(
            {"intent": "graphrag", "metadata": {"audio_b64": "voice"}}
        ) is RiskLevel.HIGH

    def test_a_malformed_risk_level_still_escalates_on_intent(self) -> None:
        assert policy_mod.resolve_risk(
            {"risk_level": "Low", "intent": "clinical"}
        ) is RiskLevel.HIGH

    def test_chitchat_without_a_risk_level_is_low(self) -> None:
        assert policy_mod.resolve_risk({"intent": "chitchat"}) is RiskLevel.LOW

    def test_an_unknown_intent_without_a_risk_level_is_standard(self) -> None:
        assert policy_mod.resolve_risk({"intent": "graphrag"}) is RiskLevel.STANDARD


class TestGuardrailsAgreeWithStreamingOnLow:
    """The two halves of the P0-1 guarantee must read LOW identically."""

    @pytest.mark.parametrize("raw", ["Low", " LOW ", "standard", "high", "bogus", None])
    def test_streaming_and_guardrails_resolve_identically(self, raw: object) -> None:
        from mao.api import streaming

        state = {"risk_level": raw}
        assert streaming.resolve_risk(state) is og.resolve_risk(state)
