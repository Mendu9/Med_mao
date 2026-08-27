"""Router tests.

Covers P0-3 (no sql route), P2-14 (chitchat detected exactly once) and the
move to registry-owned prompts.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from mao.agents import router as router_mod
from mao.agents.router import route_to_agent, router_node
from mao.graph import chitchat_gate_node
from mao.prompts import get_prompt


def _state(query: str = "what are tau tangles?") -> dict:
    return {"user_query": query, "user_id": "u1", "metadata": {}, "chat_history": []}


# ---------------------------------------------------------------------------
# P0-3 — the sql route is gone
# ---------------------------------------------------------------------------

def test_route_to_agent_has_no_sql_target() -> None:
    assert route_to_agent({"intent": "sql"}) != "sql_node"


def test_unknown_intent_falls_back_to_graphrag() -> None:
    assert route_to_agent({"intent": "sql"}) == "graphrag_node"
    assert route_to_agent({"intent": "nonsense"}) == "graphrag_node"


def test_router_module_never_names_sql_node() -> None:
    from pathlib import Path

    src = Path(router_mod.__file__).read_text(encoding="utf-8")
    assert "sql_node" not in src


@pytest.mark.parametrize(
    ("intent", "node"),
    [
        ("summarize", "summarizer_node"),
        ("graphrag", "graphrag_node"),
        ("tool", "tool_node"),
        ("multimodal", "multimodal_node"),
        ("critic", "critic_node"),
        ("clinical", "clinical_node"),
        ("chitchat", "chitchat_node"),
        ("fallback", "graphrag_node"),
    ],
)
def test_surviving_routes(intent: str, node: str) -> None:
    assert route_to_agent({"intent": intent}) == node


# ---------------------------------------------------------------------------
# P2-14 — chitchat is detected in exactly one place (the graph entry gate)
# ---------------------------------------------------------------------------

def test_chitchat_gate_is_the_detection_site() -> None:
    assert chitchat_gate_node({"user_query": "hi"})["intent"] == "chitchat"
    assert chitchat_gate_node({"user_query": "what is tau?"}).get("intent", "") != "chitchat"


def test_router_node_does_not_re_detect_chitchat() -> None:
    """The router must not carry a second deterministic chitchat detector."""
    with patch.object(router_mod, "is_chitchat") as detector, \
         patch.object(router_mod, "search_memories", return_value=""), \
         patch.object(router_mod, "_classify", return_value="graphrag"):
        router_node(_state("hi"))
    detector.assert_not_called()


def test_router_still_classifies_a_greeting_via_the_llm() -> None:
    with patch.object(router_mod, "search_memories", return_value=""), \
         patch.object(router_mod, "_classify", return_value="chitchat") as classify:
        state = router_node(_state("hi"))
    assert classify.call_count == 1
    assert state["intent"] == "chitchat"


# ---------------------------------------------------------------------------
# Deterministic attachment routing survives
# ---------------------------------------------------------------------------

def test_attachment_routes_to_clinical_without_an_llm_call() -> None:
    with patch.object(router_mod, "_classify") as classify:
        state = _state("look at this")
        state["metadata"] = {"image_b64": "abc"}
        out = router_node(state)
    classify.assert_not_called()
    assert out["intent"] == "clinical"


# ---------------------------------------------------------------------------
# Prompts come from the registry, not from inline constants
# ---------------------------------------------------------------------------

def test_router_prompts_come_from_the_registry() -> None:
    captured: dict[str, str] = {}

    def _fake_classify(system_prompt: str, user_prompt: str) -> str:
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "graphrag"

    with patch.object(router_mod, "search_memories", return_value=""), \
         patch.object(router_mod, "_classify", side_effect=_fake_classify):
        router_node(_state("what is amyloid?"))

    assert get_prompt("router.classify").template in captured["system"]
    assert "what is amyloid?" in captured["user"]


def test_router_module_holds_no_inline_prompt_constants() -> None:
    assert not hasattr(router_mod, "_ROUTER_SYSTEM")
    assert not hasattr(router_mod, "_ROUTER_USER_TEMPLATE")
