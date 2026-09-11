"""Tool agent: a ReAct loop over the capabilities declared in `mao.tools`.

The tool implementations and the dispatch table used to live here, which meant
nothing outside this module could discover what tools existed, what they cost,
or whether they were read-only. They are now declared ToolSpecs in the registry
and this module only drives the loop.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from mao import tools
from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt
from mao.prompts import get_prompt
from mao.providers import gateway
from mao.trust.classes import TrustClass
from mao.trust.egress.policy import EgressPurpose
from mao.providers.registry import ModelRole

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLS = 3

# ---------------------------------------------------------------------------
# Tool definitions (described to the LLM in the system prompt)
# ---------------------------------------------------------------------------

def _tools_description() -> str:
    """The tool block for the system prompt, derived from the registry.

    Hand-maintaining this list next to the dispatch table is how a prompt ends
    up advertising a tool that no longer exists, or hiding one that does.
    """
    return (
        "You have access to these tools. To call a tool, respond ONLY with JSON:\n"
        '{"tool": "<tool_name>", "input": "<tool_input>"}\n\n'
        "Available tools:\n"
        f"{tools.registry().describe_for_prompt()}\n\n"
        "After receiving tool output, you may call another tool OR provide a final answer.\n"
        'For a final answer respond with: {"tool": "final_answer", "input": "<your answer>"}\n'
    )


def tool_node(state: MAOState) -> MAOState:
    """
    LangGraph node: ReAct tool-calling loop using mistral.
    """
    user_query: str = state["user_query"]
    memory_context: str = state.get("memory_context", "")


    system_prompt = build_system_prompt(
        f'{get_prompt("tool.react").template}\n\n{_tools_description()}',
        memory_context,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]

    tool_trace: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    final_answer = ""

    for _turn in range(_MAX_TOOL_CALLS + 1):
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
        started = time.perf_counter()
        tool_output = tools.registry().invoke(tool_name, tool_input)
        latency_ms = (time.perf_counter() - started) * 1000
        tool_trace.append({"tool": tool_name, "input": tool_input, "output": tool_output})

        # A content-free record for the trace, alongside the ReAct transcript.
        # `TraceSchema.tool_calls` was declared and populated by nothing, so the
        # trace could not say which capability a request actually used. The two
        # lists are deliberately separate: the transcript carries tool output
        # because the model needs it, and traces are content-free by contract.
        #
        # `invoke` reports failures as its return value rather than raising —
        # that is the ReAct contract — so success is inferred from the error
        # shape it documents, not from the absence of an exception.
        failed = tool_output.startswith((f"{tool_name} error:", "Unknown tool:"))
        tool_calls.append(
            {
                "tool_id": tool_name,
                "latency_ms": round(latency_ms, 2),
                "ok": not failed,
                "error": tool_output[:200] if failed else "",
            }
        )
        logger.debug("Tool %s(%r) → %s", tool_name, tool_input, tool_output[:200])

        # Inject tool result back into conversation
        messages.append({"role": "assistant", "content": llm_response})
        messages.append({
            "role": "user",
            "content": f"Tool output for {tool_name}:\n{tool_output}\n\nContinue.",
        })

    if not final_answer:
        final_answer = "I was unable to find a complete answer with the available tools."


    state["response"]   = final_answer
    state["agent_used"] = "tool"
    state["metadata"]   = {"tool_trace": tool_trace, "tool_calls": tool_calls}
    return state


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(messages: list[dict[str, str]]) -> str:
    try:
        return gateway.complete(
            role=ModelRole.GENERAL_SYNTHESIS,
            messages=messages,
            temperature=0.0,
            purpose=EgressPurpose.GENERAL_SYNTHESIS,
            trust_class=TrustClass.SAFE_DERIVED_TEXT,
            max_tokens=512,
        ).text.strip()
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
