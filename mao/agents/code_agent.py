"""
mao/agents/code_agent.py
-------------------------
Code agent — generation, explanation, debugging, and sandboxed execution.

Route trigger: intent == "code"

When to route here:
  - "Write a Python function to merge two sorted lists"
  - "Debug this code: [paste]"
  - "Explain what this function does: [paste]"
  - "Run this script and tell me the output"
  - "Refactor this to be more Pythonic"

Design:
  - Uses llama3.1:8b for code generation (better reasoning than mistral for code)
  - Code execution: RestrictedPython + subprocess timeout sandbox
  - Three sub-modes detected from query:
      generate  — write new code
      explain   — explain existing code
      debug     — fix bugs in existing code
      execute   — run code and return output
  - Output includes the code block + explanation always

Security model for execution:
  - subprocess.run with timeout (cfg.code_exec_timeout seconds)
  - Runs in a temp directory with no network access flag
  - Only Python files executed — no shell commands
  - stdout/stderr captured and returned

Why llama3.1:8b for code:
  - Significantly better than mistral on HumanEval benchmarks
  - 8B parameters fits in ~5GB RAM — runs on consumer hardware

Libraries:
  - No extra libraries beyond stdlib for sandboxed execution
  - subprocess + tempfile + resource limits

Integration points:
  - memory/mem0_handler search before / save after
  - core/state.py       MAOState contract
  - core/config.py      code_exec_timeout
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from typing import Any

import requests
from mao.core import llm as groq_llm

from mao.core.config import cfg
from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt, save_memory, search_memories

logger = logging.getLogger(__name__)

_CODE_SYSTEM_GENERATE = """\
You are an expert software engineer. Write clean, well-commented, production-ready code.

Rules:
  - Always include type hints in Python code
  - Add a brief docstring to every function
  - Include a usage example at the bottom
  - Prefer standard library when possible; mention any pip installs required
  - Format code with consistent 4-space indentation
"""

_CODE_SYSTEM_EXPLAIN = """\
You are a patient, expert code reviewer explaining code to a developer.

Structure your explanation:
  1. What this code does (1-2 sentences)
  2. How it works step by step
  3. Notable design decisions or patterns used
  4. Potential issues or improvement areas
"""

_CODE_SYSTEM_DEBUG = """\
You are an expert debugger. Analyze the code and identify all bugs.

Structure your response:
  1. Bugs found (list each with line reference and explanation)
  2. Fixed code (complete corrected version)
  3. What was wrong (brief explanation of root cause)
"""

_CODE_SYSTEM_EXECUTE = """\
You are reviewing Python code output. Explain the results clearly.
If there was an error, explain what caused it and how to fix it.
"""

_MODE_KEYWORDS = {
    "execute": ["run", "execute", "output", "result of", "what does this print"],
    "explain": ["explain", "what does", "how does", "understand", "walk me through"],
    "debug":   ["debug", "fix", "bug", "error", "broken", "wrong", "issue", "doesn't work"],
    "generate": [],  # default
}


def code_node(state: MAOState) -> MAOState:
    """
    LangGraph node: code generation, explanation, debugging, or execution.
    """
    user_query: str = state["user_query"]
    user_id: str    = state["user_id"]
    memory_context: str = state.get("memory_context", "")

    if not memory_context:
        memory_context = search_memories(user_query, user_id)
        state["memory_context"] = memory_context

    mode = _detect_mode(user_query)
    logger.info("Code agent mode: %s", mode)

    system_map = {
        "generate": _CODE_SYSTEM_GENERATE,
        "explain":  _CODE_SYSTEM_EXPLAIN,
        "debug":    _CODE_SYSTEM_DEBUG,
        "execute":  _CODE_SYSTEM_GENERATE,  # generate then execute
    }
    system_prompt = build_system_prompt(system_map[mode], memory_context)

    # --- Generate code ---
    generated_code, full_response = _generate(system_prompt, user_query, state.get("chat_history", []))

    execution_result: dict[str, Any] = {}

    # --- Execute if requested ---
    if mode == "execute" and generated_code:
        stdout, stderr, exit_code = _safe_execute(generated_code)
        execution_result = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
        }
        # Synthesize execution result into final response
        exec_summary = _synthesize_execution(
            user_query, generated_code, stdout, stderr, exit_code
        )
        full_response = f"{full_response}\n\n**Execution output:**\n{exec_summary}"

    save_memory(user_query, full_response, user_id)

    state["response"]   = full_response
    state["agent_used"] = "code"
    state["metadata"]   = {
        "mode": mode,
        "has_code_block": bool(generated_code),
        "execution_result": execution_result,
    }
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_mode(query: str) -> str:
    """Keyword-based mode detection — fast and interpretable."""
    q_lower = query.lower()
    for mode, keywords in _MODE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return mode
    return "generate"


def _generate(
    system_prompt: str,
    user_query: str,
    chat_history: list[dict[str, str]],
) -> tuple[str, str]:
    """
    Call llama3.1:8b for code generation/explanation/debugging.

    Returns (extracted_code_block, full_llm_response).
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history[-4:])
    messages.append({"role": "user", "content": user_query})

    try:
        return groq_llm.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Execution synthesis failed: %s", exc)
        return f"stdout: {stdout}\nstderr: {stderr}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state(
        "Write a Python function that returns the nth Fibonacci number using memoization.",
        "user-test",
    )
    state = code_node(state)
    print(state["response"])
    print("Mode:", state["metadata"]["mode"])
