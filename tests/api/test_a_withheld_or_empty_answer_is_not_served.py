"""Wave 9 / B7, B8, N1 — one signal for "this is not an answer the user may keep".

Three findings, one root cause: nothing authoritative said whether a response
was clinical content or a withdrawal, so every consumer guessed separately.

  B7  an empty synthesis is served as an empty HTTP 200. The guard for this went
      into `clinical_agent` only, so `graphrag_agent` — the route real clinical
      traffic takes — still returned `""`. Downstream every control waves an
      empty response through BY DESIGN: the judge defaults to 10, the council
      records `skipped: empty_response`, and the guardrails have nothing to
      block. `00_RULES`: "Never hide failures."

  B8  a withheld answer is cached. `_cache_session` stored `result["response"]`
      unconditionally, so a transient 429 refusal was written under the query
      key for 300s. The message says "please try again in a moment" and the
      retry is served the SAME refusal from cache — the honesty fix Wave 7
      made is correct, and caching undid its only actionable half.

  N1  `apply_output_guardrails` documents "Every exit sets `output_blocked`".
      `blocked_response_node` withholds the answer without setting it, so the
      flag was false on a withheld response — a knowingly-false contract on a
      safety flag.

Fixing B7 in a second agent and B8 with a second condition would leave the
third copy to write. So `output_blocked` becomes true wherever an answer is
withheld, and the cache and the body guard both read it.
"""
from __future__ import annotations

import pytest

from mao.agents.llm_council import REVIEW_UNAVAILABLE


class TestAWithheldAnswerSaysSoInTheState:
    """N1 — the flag must mean what its docstring says."""

    def test_a_council_block_sets_output_blocked(self) -> None:
        from mao.graph import blocked_response_node

        out = blocked_response_node(
            {"council_verdict": {"passed": False, "blocked_by": "safety"}}
        )
        assert out["output_blocked"] is True
        assert out["output_blocked_by"] == "safety"

    def test_an_unavailable_review_sets_output_blocked(self) -> None:
        from mao.graph import blocked_response_node

        out = blocked_response_node(
            {"council_verdict": {"passed": False, "blocked_by": REVIEW_UNAVAILABLE}}
        )
        assert out["output_blocked"] is True
        assert out["output_blocked_by"] == REVIEW_UNAVAILABLE

    def test_the_two_reasons_still_read_differently(self) -> None:
        """Wave 7's distinction must survive: a rate limiter is not a clinical
        finding about a patient."""
        from mao.graph import blocked_response_node

        unavailable = blocked_response_node(
            {"council_verdict": {"blocked_by": REVIEW_UNAVAILABLE}}
        )["response"]
        rejected = blocked_response_node(
            {"council_verdict": {"blocked_by": "safety"}}
        )["response"]

        assert "system availability problem" in unavailable
        assert "patient safety" in rejected
        assert unavailable != rejected


class TestAWithheldAnswerIsNotCached:
    """B8 — the retry must reach the system, not a stored refusal."""

    @pytest.mark.parametrize(
        "blocked_by", [REVIEW_UNAVAILABLE, "safety", "accuracy", "hallucination"]
    )
    def test_a_withheld_response_is_not_cacheable(self, blocked_by: str) -> None:
        from mao.api.main import _may_cache_response

        assert (
            _may_cache_response(
                {"output_blocked": True, "output_blocked_by": blocked_by,
                 "response": "withheld"}
            )
            is False
        )

    def test_an_ordinary_answer_is_still_cacheable(self) -> None:
        from mao.api.main import _may_cache_response

        assert _may_cache_response(
            {"output_blocked": False, "response": "Donepezil is titrated over 4 weeks."}
        ) is True

    def test_an_empty_response_is_not_cacheable(self) -> None:
        from mao.api.main import _may_cache_response

        assert _may_cache_response({"output_blocked": False, "response": ""}) is False

    def test_a_missing_flag_is_treated_as_blocked(self) -> None:
        """Fail closed: an unknown provenance is not something to serve twice."""
        from mao.api.main import _may_cache_response

        assert _may_cache_response({"response": "text", "output_blocked": None}) is False


class TestAnEmptyBodyNeverBecomesAResponse:
    """B7 — asserted once, at the boundary, so the next agent needs no copy."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent", ["graphrag", "clinical", "summarizer", "unknown"])
    async def test_an_empty_synthesis_becomes_an_explicit_failure(
        self, agent: str
    ) -> None:
        from mao.api.finalize import finalize_response

        out = await finalize_response(
            {"response": "", "agent_used": agent, "risk_level": "standard"}, "s-1"
        )
        assert out["response"].strip(), f"{agent} produced an empty HTTP 200 body"
        assert "system failure" in out["response"].lower()

    @pytest.mark.asyncio
    async def test_a_whitespace_only_synthesis_counts_as_empty(self) -> None:
        from mao.api.finalize import finalize_response

        out = await finalize_response(
            {"response": "   \n  ", "agent_used": "graphrag"}, "s-1"
        )
        assert "system failure" in out["response"].lower()

    @pytest.mark.asyncio
    async def test_an_empty_body_is_marked_blocked_so_it_is_not_cached(self) -> None:
        """Otherwise the substituted failure message is itself cached for 300s
        and the retry is served the failure — B8's shape, one step later."""
        from mao.api.finalize import finalize_response

        out = await finalize_response({"response": "", "agent_used": "graphrag"}, "s-1")
        assert out["output_blocked"] is True

    @pytest.mark.asyncio
    async def test_a_real_answer_is_left_alone(self) -> None:
        from mao.api.finalize import finalize_response

        answer = "Donepezil is titrated from 5 mg to 10 mg after four weeks."
        out = await finalize_response(
            {"response": answer, "agent_used": "graphrag", "risk_level": "standard"},
            "s-1",
        )
        assert answer in out["response"]
        assert "system failure" not in out["response"].lower()

    @pytest.mark.asyncio
    async def test_a_pending_stream_is_not_mistaken_for_a_failure(self) -> None:
        """An agent that defers to raw-token streaming stores its prompt and
        leaves `response` empty on purpose; the body arrives from the stream."""
        from mao.api.finalize import finalize_response

        out = await finalize_response(
            {
                "response": "",
                "agent_used": "graphrag",
                "risk_level": "low",
                "intent": "chitchat",
                "_stream_messages": [{"role": "user", "content": "hello"}],
            },
            "s-1",
        )
        assert "system failure" not in out["response"].lower()


class TestTheGuardIsNotDuplicatedInTheAgents:
    """B7's actual instruction: move it OUT of the agents to one place."""

    def test_the_graphrag_agent_carries_no_local_copy(self) -> None:
        import inspect

        from mao.agents import graphrag_agent

        assert "system failure, not a clinical finding" not in inspect.getsource(
            graphrag_agent
        )

    def test_the_clinical_agent_carries_no_local_copy(self) -> None:
        import inspect

        from mao.agents import clinical_agent

        assert "system failure, not a clinical finding" not in inspect.getsource(
            clinical_agent
        )

    def test_an_empty_clinical_answer_is_not_dressed_as_a_disclaimer(self) -> None:
        """An empty body plus the mandatory disclaimer is not an answer — and it
        is non-empty, so it would slip past the boundary guard."""
        import inspect

        from mao.agents import clinical_agent

        source = inspect.getsource(clinical_agent.clinical_node)
        assert "response = response + _DISCLAIMER" not in source or (
            "if response.strip()" in source
        ), "the disclaimer is appended unconditionally, masking an empty body"
