"""GraphRAG agent tests.

Covers P0-1 (streaming must not bypass verification) and domain-aware
retrieval propagation.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from mao.agents import graphrag_agent as ga
from mao.prompts import get_prompt


class _Chunk:
    def __init__(self, text: str = "amyloid beta accumulates", score: float = 0.9) -> None:
        self.text = text
        self.score = score
        self.metadata = {"chunk_id": text[:6], "source": "s.pdf", "title": "S"}


def _state(**over) -> dict:
    base = {
        "user_query": "what is amyloid?",
        "user_id": "u1",
        "memory_context": "ctx",
        "chat_history": [],
        "sub_queries": [],
        "domain": "alzheimer",
        "metadata": {},
    }
    base.update(over)
    return base


@pytest.fixture()
def stubbed():
    with patch.object(ga, "retrieve", return_value=[_Chunk()]) as retrieve, \
         patch.object(ga, "search_memories", return_value="ctx"), \
         patch.object(ga, "save_memory"), \
         patch("mao.core.llm.chat", return_value="GENERATED ANSWER") as chat:
        yield {"retrieve": retrieve, "chat": chat}


# ---------------------------------------------------------------------------
# P0-1 — deferral is allowed only when the policy does not require verification
# ---------------------------------------------------------------------------

def test_standard_risk_generates_even_when_streaming(stubbed) -> None:
    out = ga.graphrag_node(_state(_want_stream=True, risk_level="standard"))
    assert out["response"] == "GENERATED ANSWER"
    assert not out.get("_stream_messages")


def test_high_risk_generates_even_when_streaming(stubbed) -> None:
    out = ga.graphrag_node(_state(_want_stream=True, risk_level="high"))
    assert out["response"] == "GENERATED ANSWER"
    assert not out.get("_stream_messages")


def test_missing_risk_level_generates_even_when_streaming(stubbed) -> None:
    """Fail-safe: an unset risk must never be treated as skip-verification."""
    out = ga.graphrag_node(_state(_want_stream=True))
    assert out["response"] == "GENERATED ANSWER"
    assert not out.get("_stream_messages")


def test_low_risk_may_defer_to_the_streaming_endpoint(stubbed) -> None:
    out = ga.graphrag_node(_state(_want_stream=True, risk_level="low"))
    assert out["response"] == ""
    assert out["_stream_messages"]
    assert out["_stream_model"]


def test_non_streaming_request_always_generates(stubbed) -> None:
    out = ga.graphrag_node(_state(_want_stream=False, risk_level="low"))
    assert out["response"] == "GENERATED ANSWER"
    assert not out.get("_stream_messages")


def test_deferral_never_names_a_literal_model(stubbed) -> None:
    from mao.providers.gateway import model_id_for
    from mao.providers.registry import ModelRole

    out = ga.graphrag_node(_state(_want_stream=True, risk_level="low"))
    assert out["_stream_model"] == model_id_for(ModelRole.GENERAL_SYNTHESIS)


# ---------------------------------------------------------------------------
# Domain-aware retrieval propagation
# ---------------------------------------------------------------------------

def test_single_query_retrieval_receives_the_domain(stubbed) -> None:
    ga.graphrag_node(_state(domain="stroke"))
    assert stubbed["retrieve"].call_args.kwargs["domain"] == "stroke"


def test_parallel_sub_query_retrieval_receives_the_domain(stubbed) -> None:
    ga.graphrag_node(_state(domain="stroke", sub_queries=["a?", "b?"]))
    assert stubbed["retrieve"].call_count == 2
    for call in stubbed["retrieve"].call_args_list:
        assert call.kwargs["domain"] == "stroke"


def test_missing_domain_falls_back_to_the_default(stubbed) -> None:
    state = _state()
    del state["domain"]
    ga.graphrag_node(state)
    assert stubbed["retrieve"].call_args.kwargs["domain"] == "alzheimer"


# ---------------------------------------------------------------------------
# Prompts come from the registry
# ---------------------------------------------------------------------------

def test_system_prompt_comes_from_the_registry(stubbed) -> None:
    ga.graphrag_node(_state(memory_context=""))
    system = stubbed["chat"].call_args.kwargs["messages"][0]["content"]
    assert get_prompt("graphrag.synthesis").template in system


def test_module_holds_no_inline_system_prompt() -> None:
    assert not hasattr(ga, "_GRAPHRAG_SYSTEM")
