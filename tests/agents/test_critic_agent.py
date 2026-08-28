"""Critic agent tests — P2-3: the dead ``code`` artifact branch is gone."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tests.agents.gateway_stub import _completion

from mao.agents import critic_agent as ca
from mao.prompts import get_prompt


def test_no_code_rubric_survives() -> None:
    assert not hasattr(ca, "_CRITIC_SYSTEM_CODE")


def test_no_artifact_type_keyword_table() -> None:
    assert not hasattr(ca, "_ARTIFACT_TYPE_KEYWORDS")


def test_module_source_never_mentions_a_code_intent() -> None:
    src = Path(ca.__file__).read_text(encoding="utf-8")
    assert "INTENT_CODE" not in src
    assert "code_node" not in src


def test_review_prompt_comes_from_the_registry() -> None:
    with patch("mao.providers.gateway.complete", return_value=_completion("Overall score: 7/10")) as chat:
        state = ca.critic_node(
            {"user_query": "Review this claim: tau causes AD.", "user_id": "u1"}
        )

    system = chat.call_args.kwargs["messages"][0]["content"]
    assert get_prompt("critic.review").template in system
    assert state["agent_used"] == "critic"
    assert state["metadata"]["score"] == 7.0
