"""
mao/agents/critic_agent.py
---------------------------
Critic agent — evidence-oriented review of a supplied draft, claim, or plan.

Route trigger: intent == "critic"

When to route here:
  - "Review this summary of the amyloid cascade: [paste]"
  - "What are the weaknesses in this argument?"
  - "Evaluate this study protocol"

Why a dedicated critic agent:
  - Evaluation requires a different cognitive mode than generation
  - Using a separate agent prevents the "sycophancy" trap where the same
    agent that generated content also evaluates it favorably

Design:
  - One review contract, owned by the prompt registry ("critic.review"),
    so the rubric is versioned and traceable rather than inlined here.
  - Structured output: score (1-10) plus the issues behind it.

P2-3: the code-review rubric and its artifact-type dispatch were removed with
the rest of the dead `code` intent (code_agent.py no longer exists). The
SQL-review rubric went with the SQL route (P0-3).

Integration points:
  - memory/mem0_handler search before / save after
  - core/state.py       MAOState contract
"""

from __future__ import annotations

import logging
import re
from typing import Any


from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt
from mao.prompts import get_prompt
from mao.providers import gateway
from mao.trust.egress.policy import EgressPurpose
from mao.providers.registry import ModelRole

logger = logging.getLogger(__name__)


def critic_node(state: MAOState) -> MAOState:
    """
    LangGraph node: evidence-oriented critique of a supplied draft or plan.
    """
    user_query: str = state["user_query"]
    memory_context: str = state.get("memory_context", "")


    system_prompt = build_system_prompt(
        get_prompt("critic.review").template,
        memory_context,
    )

    user_prompt = (
        f"User request: {_extract_instruction(user_query)}\n\n"
        f"Artifact to evaluate:\n{_extract_artifact(user_query)}"
    )

    response = _call_llm(system_prompt, user_prompt, state.get("chat_history", []))
    score = _extract_score(response)


    state["response"]   = response
    state["agent_used"] = "critic"
    state["metadata"]   = {"score": score}
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_instruction(query: str) -> str:
    """
    Extract the instruction part of the query (before any pasted artifact).
    Heuristic: everything before the first code fence or long paragraph.
    """
    # Find first occurrence of ``` or a very long line
    fence_pos = query.find("```")
    if fence_pos > 0:
        return query[:fence_pos].strip()
    # Long pasted content: first sentence
    sentences = query.split(".")
    return sentences[0].strip() if sentences else query[:100]


def _extract_artifact(query: str) -> str:
    """
    Extract the artifact (code block, pasted text) from the query.
    """
    # Extract code fence content
    match = re.search(r"```.*?\n(.*?)```", query, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return the full query (the LLM will handle it)
    return query


def _extract_score(response: str) -> float | None:
    """Parse 'Overall score: X/10' from the response."""
    match = re.search(r"[Oo]verall score[:\s]+(\d+(?:\.\d+)?)\s*/\s*10", response)
    if match:
        return float(match.group(1))
    return None


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    chat_history: list[dict[str, str]],
) -> str:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    # De-identified upstream, at the protected input boundary, not here. This
    # was the third of the three sites where `state["chat_history"]` reached a
    # provider with the scrubber never having been called on it (ADV15-8).
    messages.extend(chat_history[-4:])
    messages.append({"role": "user", "content": user_prompt})

    try:
        return gateway.complete(
            role=ModelRole.GENERAL_SYNTHESIS,
            messages=messages,
            temperature=0.2,
            purpose=EgressPurpose.SAFETY_VERIFICATION,
            max_tokens=768,
        ).text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Critic LLM call failed: %s", exc)
        return f"Critique generation failed: {exc}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state(
        "Review this claim: amyloid plaques alone are sufficient to cause dementia.",
        "user-test",
    )
    state = critic_node(state)
    print(state["response"])
    print("Score:", state["metadata"].get("score"))
