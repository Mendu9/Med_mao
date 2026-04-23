"""
mao/agents/critic_agent.py
---------------------------
Critic agent — evaluation, review, and feedback on text, code, or plans.

Route trigger: intent == "critic"

When to route here:
  - "Review my essay: [paste]"
  - "Is this code well-written? [paste]"
  - "Evaluate this business plan"
  - "What are the weaknesses in this argument?"
  - "Give me feedback on my answer to this interview question"
  - "Score this SQL query for performance"

Why a dedicated critic agent:
  - Evaluation requires a different cognitive mode than generation
  - Using a separate agent prevents the "sycophancy" trap where the same
    agent that generated content also evaluates it favorably
  - Structured rubric-based evaluation is more defensible and actionable
  - Interview-ready argument: separating critic from generator is a known
    technique in Constitutional AI and RLHF pipelines

Design:
  - Detects artifact type: code, text/prose, SQL, plan/outline
  - Applies appropriate rubric for each type
  - Structured output: score (1-10), strengths, weaknesses, specific suggestions
  - Uses llama3.1:8b — deeper reasoning produces better critique

Integration points:
  - memory/mem0_handler search before / save after
  - core/state.py       MAOState contract
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests
from mao.core import llm as groq_llm

from mao.core.config import cfg
from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt, save_memory, search_memories

logger = logging.getLogger(__name__)

_CRITIC_SYSTEM_CODE = """\
You are a senior software engineer conducting a code review.

Evaluate the code on these dimensions:
  1. Correctness     — does it do what it claims? Any bugs?
  2. Readability     — naming, comments, structure
  3. Efficiency      — time/space complexity, unnecessary operations
  4. Security        — injection risks, unsafe operations, hardcoded secrets
  5. Pythonic style  — idiomatic use of the language
  6. Testability     — is it easy to unit test?

Output format:
  Overall score: X/10

  Strengths:
  - ...

  Issues (severity: high/medium/low):
  - [HIGH] ...
  - [MED]  ...

  Specific suggestions:
  - ...
"""

_CRITIC_SYSTEM_PROSE = """\
You are an expert editor providing constructive feedback on written text.

Evaluate on:
  1. Clarity        — is the message clear to the target audience?
  2. Structure      — logical flow, transitions, paragraphing
  3. Argumentation  — are claims supported with evidence?
  4. Conciseness    — unnecessary wordiness
  5. Tone           — appropriate for the context?

Output format:
  Overall score: X/10

  Strengths:
  - ...

  Areas for improvement:
  - ...

  Specific line-level suggestions (quote the original, then suggest):
  - Original: "..."
    Suggestion: "..."
"""

_CRITIC_SYSTEM_SQL = """\
You are a database performance expert reviewing a SQL query.

Evaluate on:
  1. Correctness      — will it return the expected results?
  2. Performance      — missing indexes, N+1, full table scans, unnecessary subqueries
  3. Readability      — aliasing, formatting, naming
  4. Safety           — SQL injection risk (if parameterization is absent)
  5. Edge cases       — NULL handling, empty sets

Output format:
  Overall score: X/10

  Correctness verdict: PASS / FAIL / UNSURE
  Performance concerns: (list)
  Suggested rewrite: (only if substantially different)
"""

_CRITIC_SYSTEM_PLAN = """\
You are a senior consultant evaluating a plan or proposal.

Evaluate on:
  1. Feasibility     — is this achievable with stated constraints?
  2. Completeness    — are there gaps or unstated assumptions?
  3. Risk            — what could go wrong? Is risk mitigation present?
  4. Clarity         — is the success criterion measurable?
  5. Alternatives    — were better approaches overlooked?

Output format:
  Overall score: X/10

  What works:
  - ...

  Risks and gaps:
  - [CRITICAL] ...
  - [MINOR]    ...

  Recommendations:
  - ...
"""

_ARTIFACT_TYPE_KEYWORDS = {
    "code": ["def ", "class ", "import ", "function", "python", "javascript", "```"],
    "sql":  ["select ", "from ", "where ", "join ", "group by", "sql"],
    "plan": ["plan", "proposal", "roadmap", "strategy", "outline", "steps to"],
}


def critic_node(state: MAOState) -> MAOState:
    """
    LangGraph node: structured critique of code, prose, SQL, or plans.
    """
    user_query: str = state["user_query"]
    user_id: str    = state["user_id"]
    memory_context: str = state.get("memory_context", "")

    if not memory_context:
        memory_context = search_memories(user_query, user_id)
        state["memory_context"] = memory_context

    artifact_type = _detect_artifact_type(user_query)
    logger.info("Critic agent artifact type: %s", artifact_type)

    system_map = {
        "code":  _CRITIC_SYSTEM_CODE,
        "sql":   _CRITIC_SYSTEM_SQL,
        "plan":  _CRITIC_SYSTEM_PLAN,
        "prose": _CRITIC_SYSTEM_PROSE,
    }
    system_prompt = build_system_prompt(
        system_map.get(artifact_type, _CRITIC_SYSTEM_PROSE),
        memory_context,
    )

    user_prompt = (
        f"User request: {_extract_instruction(user_query)}\n\n"
        f"Artifact to evaluate:\n{_extract_artifact(user_query)}"
    )

    response = _call_llm(system_prompt, user_prompt, state.get("chat_history", []))
    score = _extract_score(response)

    save_memory(user_query, response, user_id)

    state["response"]   = response
    state["agent_used"] = "critic"
    state["metadata"]   = {
        "artifact_type": artifact_type,
        "score": score,
    }
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_artifact_type(query: str) -> str:
    q_lower = query.lower()
    for atype, keywords in _ARTIFACT_TYPE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return atype
    return "prose"


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
    messages.extend(chat_history[-4:])
    messages.append({"role": "user", "content": user_prompt})

    try:
        return groq_llm.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=768,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Critic LLM call failed: %s", exc)
        return f"Critique generation failed: {exc}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state(
        "Review this code:\n```python\ndef add(a, b):\n    return a+b\n```",
        "user-test",
    )
    state = critic_node(state)
    print(state["response"])
    print("Score:", state["metadata"].get("score"))
