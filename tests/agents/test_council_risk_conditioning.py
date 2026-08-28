"""arch-M3 — the council must be risk-conditioned, like the rest of verification.

The architecture asks for "risk-conditioned verification instead of a full
council on every query". `verification_node` complies. The council did not:
`graph.py` wired `domain_supervisor -> council` unconditionally and
`llm_council.py` contained no reference to risk or policy at all, so a LOW-risk
chitchat the policy exempts from verification still paid a SAFETY_JUDGE call.

Narrowing on *context emptiness* is not the same thing as narrowing on risk.

adv-L1 — `route_to_agent` silently mapped every unrecognised intent to
`graphrag_node`, the one route that carries no clinical disclaimer.
"""
from __future__ import annotations

import pytest

from mao.agents.llm_council import council_node
from mao.safety.policy import RiskLevel


def _state(**over) -> dict:
    base = {
        "response": "Amyloid plaques accumulate in the cortex.",
        "retrieved_docs": ["some context"],
        "risk_level": RiskLevel.STANDARD.value,
    }
    base.update(over)
    return base


class TestTheCouncilSkipsWhatThePolicyExempts:
    def test_low_risk_does_not_call_a_judge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            "mao.agents.llm_council.run_council",
            lambda **kwargs: called.append("ran") or {"passed": True},
        )
        out = council_node(_state(risk_level=RiskLevel.LOW.value))
        assert called == []
        assert out["council_verdict"]["passed"] is True
        assert out["council_verdict"]["skipped"] == "low_risk"

    @pytest.mark.parametrize(
        "risk", [RiskLevel.STANDARD.value, RiskLevel.HIGH.value]
    )
    def test_verified_risk_levels_still_run_the_council(
        self, risk: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[str] = []

        def _run(**kwargs: object) -> dict:
            called.append("ran")
            return {"passed": True, "blocked_by": None}

        monkeypatch.setattr("mao.agents.llm_council.run_council", _run)
        council_node(_state(risk_level=risk))
        assert called == ["ran"]

    def test_a_malformed_risk_level_still_runs_the_council(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail closed: an unreadable risk must not buy an exemption."""
        called: list[str] = []

        def _run(**kwargs: object) -> dict:
            called.append("ran")
            return {"passed": True, "blocked_by": None}

        monkeypatch.setattr("mao.agents.llm_council.run_council", _run)
        council_node(_state(risk_level="Low"))
        assert called == ["ran"]

    def test_a_missing_risk_level_still_runs_the_council(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[str] = []

        def _run(**kwargs: object) -> dict:
            called.append("ran")
            return {"passed": True, "blocked_by": None}

        monkeypatch.setattr("mao.agents.llm_council.run_council", _run)
        state = _state()
        del state["risk_level"]
        council_node(state)
        assert called == ["ran"]

    def test_the_council_consults_the_policy(self) -> None:
        import inspect

        from mao.agents import llm_council

        source = inspect.getsource(llm_council)
        assert "policy" in source or "requires_verification" in source

    def test_an_empty_response_still_short_circuits(self) -> None:
        out = council_node(_state(response="   "))
        assert out["council_verdict"]["skipped"] == "empty_response"


class TestUnrecognisedIntentsDoNotSilentlyBecomeGraphrag:
    """adv-L1 — the fallback route carries no clinical disclaimer."""

    @pytest.mark.parametrize("intent", ["CLINICAL", "clinical ", "", None, "banana"])
    def test_an_unrecognised_intent_is_refused(self, intent: object) -> None:
        from mao.agents.router import route_to_agent

        with pytest.raises(ValueError):
            route_to_agent({"intent": intent})

    @pytest.mark.parametrize(
        ("intent", "node"),
        [
            ("graphrag", "graphrag_node"),
            ("clinical", "clinical_node"),
            ("chitchat", "chitchat_node"),
            ("fallback", "graphrag_node"),
        ],
    )
    def test_known_intents_still_route(self, intent: str, node: str) -> None:
        from mao.agents.router import route_to_agent

        assert route_to_agent({"intent": intent}) == node

    def test_a_missing_intent_uses_the_declared_fallback(self) -> None:
        from mao.agents.router import route_to_agent

        assert route_to_agent({}) == "graphrag_node"
