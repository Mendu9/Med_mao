"""Tests for senior_supervisor_node — completeness check, independent of transport.

P0-1: this node used to skip its completeness check whenever `_want_stream` was
set and the answer was still empty. The previous version of this file asserted
that skip as correct behaviour. It is not: it made supervision a function of the
transport, so a streamed request was reviewed by nobody. What legitimately skips
the LLM call is having no sub-questions, or having no answer — and an empty
answer answers none of its sub-questions, which is a conclusion we can reach
without spending a 70B call on it.
"""
from __future__ import annotations

import json

import pytest

from mao.agents.senior_supervisor import senior_supervisor_node
from mao.providers import gateway
from mao.providers.llm.base import ProviderResponse

_SUB_QUERIES = ["What is CBF?", "What is CBV?"]


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = "{}", raises: bool = False) -> None:
        self.text = text
        self.raises = raises
        self.calls: list[dict] = []

    def complete(self, *, model_id, messages, temperature, max_tokens) -> ProviderResponse:
        if self.raises:
            raise RuntimeError("provider down")
        self.calls.append({"messages": messages})
        return ProviderResponse(text=self.text)


@pytest.fixture
def bind_provider():
    def _bind(text: str = "{}", raises: bool = False) -> FakeProvider:
        provider = FakeProvider(text, raises=raises)
        gateway.set_provider(provider)
        return provider

    yield _bind
    gateway.reset_provider()


def test_no_sub_queries_short_circuits_without_llm(bind_provider):
    """With no sub_queries there is nothing to check."""
    provider = bind_provider()
    out = senior_supervisor_node({"sub_queries": [], "response": ""})

    assert provider.calls == []
    assert out["completeness_ok"] is True
    assert out["missing_sub_queries"] == []


def test_answered_sub_queries_report_complete(bind_provider):
    provider = bind_provider(json.dumps({"answered": _SUB_QUERIES, "missing": []}))

    out = senior_supervisor_node({
        "sub_queries": _SUB_QUERIES,
        "response": "CBF is cerebral blood flow. CBV is cerebral blood volume.",
    })

    assert len(provider.calls) == 1
    assert out["completeness_ok"] is True
    assert out["missing_sub_queries"] == []


def test_missing_sub_queries_report_incomplete(bind_provider):
    bind_provider(json.dumps({"answered": ["What is CBF?"], "missing": ["What is CBV?"]}))

    out = senior_supervisor_node({
        "sub_queries": _SUB_QUERIES,
        "response": "CBF is cerebral blood flow.",
    })

    assert out["completeness_ok"] is False
    assert out["missing_sub_queries"] == ["What is CBV?"]


def test_uses_the_registered_completeness_prompt(bind_provider):
    from mao.prompts import get_prompt

    provider = bind_provider(json.dumps({"missing": []}))
    senior_supervisor_node({"sub_queries": _SUB_QUERIES, "response": "An answer."})

    spec = get_prompt("senior_supervisor.completeness")
    assert provider.calls[0]["messages"][0]["content"] == spec.template


@pytest.mark.parametrize("bad", ["not json", "", "{broken", "[]"])
def test_malformed_output_does_not_raise(bind_provider, bad):
    bind_provider(bad)
    out = senior_supervisor_node({"sub_queries": _SUB_QUERIES, "response": "An answer."})
    assert out["missing_sub_queries"] == []


def test_provider_failure_does_not_raise(bind_provider):
    bind_provider(raises=True)
    out = senior_supervisor_node({"sub_queries": _SUB_QUERIES, "response": "An answer."})
    assert isinstance(out["missing_sub_queries"], list)


# ---------------------------------------------------------------------------
# P0-1 — supervision must not depend on the transport
# ---------------------------------------------------------------------------

def test_streaming_request_with_an_answer_is_still_checked(bind_provider):
    """This is the case the old short-circuit could not distinguish: `_want_stream`
    is set AND there is a real answer. It must be checked like any other."""
    provider = bind_provider(json.dumps({"answered": [], "missing": ["What is CBV?"]}))

    out = senior_supervisor_node({
        "sub_queries": _SUB_QUERIES,
        "response": "CBF is cerebral blood flow.",
        "_want_stream": True,
    })

    assert len(provider.calls) == 1, "a streamed answer must still be checked"
    assert out["completeness_ok"] is False
    assert out["missing_sub_queries"] == ["What is CBV?"]


def test_streamed_and_buffered_states_produce_the_same_result(bind_provider):
    bind_provider(json.dumps({"missing": ["What is CBV?"]}))

    streamed = senior_supervisor_node({
        "sub_queries": _SUB_QUERIES, "response": "An answer.", "_want_stream": True,
    })
    buffered = senior_supervisor_node({
        "sub_queries": _SUB_QUERIES, "response": "An answer.",
    })

    assert streamed["completeness_ok"] == buffered["completeness_ok"]
    assert streamed["missing_sub_queries"] == buffered["missing_sub_queries"]


def test_empty_answer_is_incomplete_without_spending_an_llm_call(bind_provider):
    """An empty answer answers none of its sub-questions. That is a conclusion,
    not a pass — and it needs no model call to reach."""
    provider = bind_provider()

    out = senior_supervisor_node({"sub_queries": _SUB_QUERIES, "response": ""})

    assert provider.calls == [], "no model call for an empty answer"
    assert out["completeness_ok"] is False
    assert out["missing_sub_queries"] == _SUB_QUERIES


def test_empty_answer_result_is_the_same_with_or_without_the_stream_flag(bind_provider):
    """The old code returned completeness_ok=True here purely because
    `_want_stream` was set. Transport must make no difference."""
    bind_provider()

    streamed = senior_supervisor_node({
        "sub_queries": _SUB_QUERIES, "response": "", "_want_stream": True,
    })
    buffered = senior_supervisor_node({"sub_queries": _SUB_QUERIES, "response": ""})

    assert streamed["completeness_ok"] == buffered["completeness_ok"]
    assert streamed["missing_sub_queries"] == buffered["missing_sub_queries"]
