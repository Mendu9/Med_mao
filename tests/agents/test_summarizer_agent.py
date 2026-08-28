"""Summarizer tests — domain propagation and registry-owned prompts."""
from __future__ import annotations

from unittest.mock import patch

from mao.agents import summarizer_agent as sa
from mao.prompts import get_prompt


class _Chunk:
    text = "ischemic stroke is caused by an occlusion"


def test_kb_retrieval_receives_the_domain() -> None:
    with patch.object(sa, "retrieve", return_value=[_Chunk()]) as retrieve, \
         patch("mao.core.llm.chat", return_value="summary"):
        sa.summarizer_node(
            {"user_query": "summarise stroke care", "user_id": "u1", "domain": "stroke"}
        )

    assert retrieve.call_args.kwargs["domain"] == "stroke"


def test_summarizer_prompt_comes_from_the_registry() -> None:
    with patch.object(sa, "retrieve", return_value=[_Chunk()]), \
         patch("mao.core.llm.chat", return_value="summary") as chat:
        sa.summarizer_node({"user_query": "summarise stroke care", "user_id": "u1"})

    system = chat.call_args.kwargs["messages"][0]["content"]
    assert get_prompt("summarizer.synthesis").template in system


def test_module_holds_no_inline_summary_system_prompt() -> None:
    assert not hasattr(sa, "_SUMMARIZE_SYSTEM")
