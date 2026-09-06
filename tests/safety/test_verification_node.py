"""The verification node is what makes the output guardrails reachable.

P1-1  `judge_scores` was never written to graph state, so the judge BLOCK and
      WARN branches in `output_guardrails` were dead code in production.
P1-2  `nli_flags` was written only by `clinical_agent`, so the NLI entailment
      gate was inert for every other agent.

Both are fixed by computing them in one agent-independent place. These tests
are hermetic: the judge runs against a fake provider bound to the gateway, and
the NLI checker is stubbed, so nothing touches the network or loads a model.
"""
from __future__ import annotations

import json

import pytest

from mao.providers import gateway
from mao.providers.llm.base import ProviderResponse
from mao.safety.policy import RiskLevel, get_policy
from mao.safety.verification import verification_node

_RESPONSE = (
    "Donepezil is an acetylcholinesterase inhibitor used in mild Alzheimer's disease. "
    "It is titrated from five milligrams to ten milligrams daily over four weeks. "
    "Gastrointestinal side effects are the most common reason for discontinuation."
)

_JUDGE_JSON = json.dumps({"safety": 8, "groundedness": 7, "notes": "Well grounded."})


class JudgeFakeProvider:
    name = "judge-fake"

    def __init__(self, text: str = _JUDGE_JSON, raises: bool = False) -> None:
        self.text = text
        self.raises = raises
        self.calls: list[dict] = []

    def complete(self, *, model_id, messages, temperature, max_tokens) -> ProviderResponse:
        if self.raises:
            raise RuntimeError("provider down")
        self.calls.append({"model_id": model_id, "messages": messages})
        return ProviderResponse(text=self.text)


@pytest.fixture
def bind_judge():
    def _bind(text: str = _JUDGE_JSON, raises: bool = False) -> JudgeFakeProvider:
        provider = JudgeFakeProvider(text, raises=raises)
        gateway.set_provider(provider)
        return provider

    yield _bind
    gateway.reset_provider()


@pytest.fixture
def stub_nli(monkeypatch):
    """Deterministic NLI so these tests never load a cross-encoder."""
    calls: list[dict] = []

    def _fake(claims, premise):
        calls.append({"claims": claims, "premise": premise})
        return [{"claim": c, "entailed": i % 2 == 0, "score": 0.9} for i, c in enumerate(claims)]

    monkeypatch.setattr("mao.safety.verification.check_all_claims", _fake)
    return calls


def _state(**overrides) -> dict:
    base = {
        "response": _RESPONSE,
        "retrieved_docs": ["Donepezil inhibits acetylcholinesterase in the brain."],
        "risk_level": "high",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_writes_both_keys_the_guardrails_read(bind_judge, stub_nli):
    bind_judge()
    out = verification_node(_state())

    assert "nli_flags" in out
    assert "judge_scores" in out


def test_nli_flags_shape(bind_judge, stub_nli):
    bind_judge()
    out = verification_node(_state())

    assert isinstance(out["nli_flags"], list)
    assert out["nli_flags"], "a response with claims and context must produce flags"
    for flag in out["nli_flags"]:
        assert isinstance(flag, dict)
        assert isinstance(flag["entailed"], bool)


def test_judge_scores_shape(bind_judge, stub_nli):
    bind_judge()
    scores = verification_node(_state())["judge_scores"]

    assert isinstance(scores["safety"], int)
    assert isinstance(scores["groundedness"], int)
    assert isinstance(scores["notes"], str)
    assert 0 <= scores["safety"] <= 10
    assert 0 <= scores["groundedness"] <= 10


def test_judge_scores_carry_the_model_verdict(bind_judge, stub_nli):
    bind_judge()
    scores = verification_node(_state())["judge_scores"]

    assert scores["safety"] == 8
    assert scores["groundedness"] == 7


def test_returns_state_plus_results(bind_judge, stub_nli):
    bind_judge()
    state = _state(user_id="u-1", agent_used="graphrag")
    out = verification_node(state)

    assert out["user_id"] == "u-1"
    assert out["agent_used"] == "graphrag"
    assert out["response"] == _RESPONSE, "verification observes, it does not edit"


def test_uses_the_registered_judge_prompt_and_the_safety_judge_role(bind_judge, stub_nli):
    from mao.prompts import get_prompt
    from mao.providers.registry import ModelRole

    provider = bind_judge()
    verification_node(_state())

    spec = get_prompt("judge.safety")
    messages = provider.calls[0]["messages"]
    # Was `== spec.template`. Superseded by ADV15-10: the judge's system turn is
    # now its registered template PLUS the non-forgeable protocol framing, and
    # the evidence and the answer arrive as separate messages instead of one
    # concatenated user turn. Equality here would forbid the fix; the property
    # that matters is still that the call site sends the REGISTERED prompt and
    # not inline text of its own.
    assert messages[0]["content"].startswith(spec.template)
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert "RESPONSE_UNDER_REVIEW" in messages[-1]["content"]
    assert provider.calls[0]["model_id"] == gateway.model_id_for(ModelRole.SAFETY_JUDGE)


# ---------------------------------------------------------------------------
# P1-2 — the NLI gate must work for every agent, not just clinical
# ---------------------------------------------------------------------------

def test_nli_runs_for_a_non_clinical_agent(bind_judge, stub_nli):
    """The gate used to be inert unless clinical_agent happened to run."""
    bind_judge()
    out = verification_node(_state(agent_used="graphrag"))

    assert out["nli_flags"], "NLI must not depend on which agent produced the answer"
    assert stub_nli, "the NLI checker must actually have been called"


def test_nli_premise_comes_from_retrieved_docs(bind_judge, stub_nli):
    bind_judge()
    verification_node(_state(retrieved_docs=["alpha premise text", "beta premise text"]))

    assert "alpha premise text" in stub_nli[0]["premise"]


def test_no_retrieved_docs_yields_no_flags(bind_judge, stub_nli):
    """With nothing to entail against, an empty flag list is honest; an
    all-unentailed list would block every un-retrieved answer."""
    bind_judge()
    out = verification_node(_state(retrieved_docs=[]))

    assert out["nli_flags"] == []
    assert stub_nli == [], "no premise means no NLI call"


def test_retrieved_docs_objects_are_supported(bind_judge, stub_nli):
    """Retrievers return chunk objects, not strings."""
    class Chunk:
        text = "Donepezil inhibits acetylcholinesterase."

    bind_judge()
    verification_node(_state(retrieved_docs=[Chunk()]))

    assert "acetylcholinesterase" in stub_nli[0]["premise"]


# ---------------------------------------------------------------------------
# Policy — the judge is skipped only where policy says verification is optional
# ---------------------------------------------------------------------------

def test_low_risk_skips_the_llm_judge(bind_judge, stub_nli):
    assert get_policy().requires_verification(RiskLevel.LOW) is False
    provider = bind_judge()

    out = verification_node(_state(risk_level="low"))

    assert provider.calls == [], "chitchat must not pay for a judge call"
    assert isinstance(out["judge_scores"], dict)


@pytest.mark.parametrize("risk", ["standard", "high"])
def test_verified_risk_levels_run_the_judge(bind_judge, stub_nli, risk):
    provider = bind_judge()
    verification_node(_state(risk_level=risk))
    assert len(provider.calls) == 1


def test_risk_level_accepts_the_enum(bind_judge, stub_nli):
    provider = bind_judge()
    verification_node(_state(risk_level=RiskLevel.HIGH))
    assert len(provider.calls) == 1


def test_missing_risk_level_defaults_from_intent(bind_judge, stub_nli):
    """No risk_level on state: fall back to the policy's own classification."""
    provider = bind_judge()
    state = _state(intent="clinical")
    del state["risk_level"]

    verification_node(state)

    assert len(provider.calls) == 1


def test_missing_risk_level_for_chitchat_skips_the_judge(bind_judge, stub_nli):
    provider = bind_judge()
    state = _state(intent="chitchat")
    del state["risk_level"]

    verification_node(state)

    assert provider.calls == []


# ---------------------------------------------------------------------------
# "Must never raise"
# ---------------------------------------------------------------------------

def test_empty_response_verifies_nothing(bind_judge, stub_nli):
    provider = bind_judge()
    out = verification_node(_state(response=""))

    assert out["nli_flags"] == []
    assert provider.calls == []
    assert isinstance(out["judge_scores"], dict)


def test_provider_failure_does_not_raise(bind_judge, stub_nli):
    bind_judge(raises=True)
    out = verification_node(_state())

    assert isinstance(out["judge_scores"]["safety"], int)
    assert isinstance(out["judge_scores"]["notes"], str)


def test_nli_failure_does_not_raise(bind_judge, monkeypatch):
    bind_judge()

    def _boom(claims, premise):
        raise RuntimeError("cross-encoder exploded")

    monkeypatch.setattr("mao.safety.verification.check_all_claims", _boom)
    out = verification_node(_state())

    assert out["nli_flags"] == []


@pytest.mark.parametrize("bad_judge_output", ["not json", "", "{broken", "[]", "null"])
def test_malformed_judge_output_still_yields_a_valid_dict(bind_judge, stub_nli, bad_judge_output):
    bind_judge(bad_judge_output)
    scores = verification_node(_state())["judge_scores"]

    assert isinstance(scores["safety"], int)
    assert isinstance(scores["groundedness"], int)
    assert isinstance(scores["notes"], str)


def test_judge_output_in_a_markdown_fence_is_parsed(bind_judge, stub_nli):
    bind_judge(f"```json\n{_JUDGE_JSON}\n```")
    scores = verification_node(_state())["judge_scores"]
    assert scores["safety"] == 8


def test_out_of_range_judge_scores_are_clamped(bind_judge, stub_nli):
    bind_judge(json.dumps({"safety": 99, "groundedness": -4, "notes": "odd"}))
    scores = verification_node(_state())["judge_scores"]

    assert scores["safety"] == 10
    assert scores["groundedness"] == 0


@pytest.mark.parametrize("state", [
    {},
    {"response": None},
    {"response": _RESPONSE, "retrieved_docs": None},
    {"response": _RESPONSE, "risk_level": "not-a-risk-level"},
    {"response": _RESPONSE, "retrieved_docs": [None, 42]},
])
def test_degenerate_state_does_not_raise(bind_judge, stub_nli, state):
    bind_judge()
    out = verification_node(state)

    assert isinstance(out["nli_flags"], list)
    assert isinstance(out["judge_scores"], dict)
