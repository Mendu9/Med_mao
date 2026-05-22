"""
mao/agents/tool_agent.py
-------------------------
Tool agent — web search, calculator, Wikipedia API.

Route trigger: intent == "tool"

When to route here:
  - "Search the web for the latest news on X"
  - "What is 15% of 847?"
  - "Look up the Wikipedia page for quantum computing"
  - "What's the current population of Brazil?"
  - Any query that requires live/real-time data (not in the KB)

Tools available:
  1. DuckDuckGoSearch  — live web search via duckduckgo-search library
  2. WikipediaAPI      — Wikipedia summary + full sections via wikipedia-api
  3. Calculator        — safe Python eval via numexpr (no arbitrary code exec)

Design:
  - LLM (mistral) decides which tool(s) to call via a ReAct-style loop
  - Max 3 tool calls per turn to prevent infinite loops
  - Tool results injected back into prompt for final synthesis
  - Uses mistral for speed (tool selection is simple classification)

Libraries:
  - duckduckgo_search  (pip install duckduckgo-search)
  - wikipedia-api      (pip install wikipedia-api)
  - numexpr            (pip install numexpr)

Integration points:
  - memory/mem0_handler search before / save after
  - core/state.py       MAOState contract
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numexpr
import requests
from mao.core import llm as groq_llm
import wikipediaapi

from mao.core.web_search import web_search as _web_search_provider

from mao.core.config import cfg
from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt, save_memory, search_memories

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLS = 3

# ---------------------------------------------------------------------------
# Tool definitions (described to the LLM in the system prompt)
# ---------------------------------------------------------------------------

_TOOLS_DESCRIPTION = """\
You have access to these tools. To call a tool, respond ONLY with JSON:
{"tool": "<tool_name>", "input": "<tool_input>"}

Available tools:
  web_search(query)     - Search the web for current information. Input: search query string.
  wikipedia(topic)      - Get a Wikipedia summary for a topic. Input: topic name string.
  calculator(expression)- Evaluate a math expression. Input: valid math expression (e.g. "15 * 0.15").

After receiving tool output, you may call another tool OR provide a final answer.
For a final answer respond with: {"tool": "final_answer", "input": "<your answer>"}
"""

_TOOL_AGENT_SYSTEM = """\
You are a helpful assistant that uses tools to answer questions accurately.
Never guess when a tool can give you the real answer.
Always verify numbers with the calculator tool before stating them.
"""


def tool_node(state: MAOState) -> MAOState:
    """
    LangGraph node: ReAct tool-calling loop using mistral.
    """
    user_query: str = state["user_query"]
    user_id: str    = state["user_id"]
    memory_context: str = state.get("memory_context", "")

    if not memory_context:
        memory_context = search_memories(user_query, user_id)
        state["memory_context"] = memory_context

    system_prompt = build_system_prompt(
        f"{_TOOL_AGENT_SYSTEM}\n\n{_TOOLS_DESCRIPTION}",
        memory_context,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]

    tool_trace: list[dict[str, Any]] = []
    final_answer = ""

    for turn in range(_MAX_TOOL_CALLS + 1):
        llm_response = _call_llm(messages)
        parsed = _parse_tool_call(llm_response)

        if parsed is None:
            # LLM gave a plain text response — treat as final answer
            final_answer = llm_response
            break

        tool_name  = parsed.get("tool", "")
        tool_input = parsed.get("input", "")

        if tool_name == "final_answer":
            final_answer = tool_input
            break

        # Execute tool
        tool_output = _dispatch_tool(tool_name, tool_input)
        tool_trace.append({"tool": tool_name, "input": tool_input, "output": tool_output})
        logger.debug("Tool %s(%r) → %s", tool_name, tool_input, tool_output[:200])

        # Inject tool result back into conversation
        messages.append({"role": "assistant", "content": llm_response})
        messages.append({
            "role": "user",
            "content": f"Tool output for {tool_name}:\n{tool_output}\n\nContinue.",
        })

    if not final_answer:
        final_answer = "I was unable to find a complete answer with the available tools."

    save_memory(user_query, final_answer, user_id)

    state["response"]   = final_answer
    state["agent_used"] = "tool"
    state["metadata"]   = {"tool_trace": tool_trace}
    return state


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _web_search(query: str) -> str:
    """Web search — returns top 5 results as formatted text via multi-provider fallback."""
    results = _web_search_provider(query, num_results=5)
    if not results:
        return "No web results found."
    lines = []
    for r in results:
        title = r.get("title", "")
        body  = r.get("body", "")[:300]
        href  = r.get("href", "")
        lines.append(f"- {title}\n  {body}\n  URL: {href}")
    return "\n\n".join(lines)


def _wikipedia_lookup(topic: str) -> str:
    """Fetch Wikipedia summary for a topic (English)."""
    try:
        wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="MAO-Agent/1.0 (https://github.com/mao-project)",
        )
        page = wiki.page(topic)
        if not page.exists():
            return f"No Wikipedia page found for '{topic}'."
        # Return first 1500 chars of summary
        return page.summary[:1500]
    except Exception as exc:  # noqa: BLE001
        logger.error("Wikipedia lookup failed: %s", exc)
        return f"Wikipedia error: {exc}"


def _calculator(expression: str) -> str:
    """
    Safe math evaluation via numexpr.

    numexpr supports basic arithmetic, trig, log — no arbitrary Python execution.
    Intentionally NOT using eval() — that would be a security vulnerability.
    """
    try:
        # Strip any markdown code fences
        clean_expr = re.sub(r"```.*?```", "", expression, flags=re.DOTALL).strip()
        result = numexpr.evaluate(clean_expr)
        return str(float(result))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Calculator error for '%s': %s", expression, exc)
        return f"Calculator error: {exc}"


def _dispatch_tool(tool_name: str, tool_input: str) -> str:
    dispatch: dict[str, Any] = {
        "web_search":  _web_search,
        "wikipedia":   _wikipedia_lookup,
        "calculator":  _calculator,
    }
    fn = dispatch.get(tool_name)
    if fn is None:
        return f"Unknown tool: {tool_name}"
    return fn(tool_input)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(messages: list[dict[str, str]]) -> str:
    try:
        return groq_llm.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Tool agent LLM call failed: %s", exc)
        return '{"tool": "final_answer", "input": "LLM error, cannot complete request."}'


def _parse_tool_call(text: str) -> dict[str, str] | None:
    """Extract JSON tool call from LLM response, or return None if plain text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state("What is 12% of 3750?", "user-test")
    state = tool_node(state)
    print(state["response"])
    print("Tool trace:", state["metadata"]["tool_trace"])
