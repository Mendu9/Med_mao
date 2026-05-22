"""
mao/core/groq_usage.py — Thread-safe Groq API usage tracker.

Tracks token consumption, estimated cost, and rate-limit headers across all
LLM calls made within a process lifetime. A module-level singleton `tracker`
is imported by llm.py so every call site records automatically.

Usage:
    from mao.core.groq_usage import tracker
    tracker.record(model="llama-3.1-8b-instant", input_tokens=120, output_tokens=80)
    summary = tracker.get_summary()
    tracker.reset()
"""
from __future__ import annotations

import threading
from typing import Any


# ---------------------------------------------------------------------------
# Pricing — $ per million tokens (as of May 2026)
# https://console.groq.com/docs/openai
# ---------------------------------------------------------------------------
_PRICING: dict[str, dict[str, float]] = {
    "llama-3.1-8b-instant":    {"input": 0.05,  "output": 0.10},
    "llama-3.3-70b-versatile": {"input": 0.59,  "output": 0.79},
    "default":                  {"input": 0.05,  "output": 0.10},
}


class GroqUsageTracker:
    """Thread-safe accumulator for Groq API token usage and rate-limit state.

    All public methods are safe to call from multiple threads simultaneously.
    The singleton `tracker` at module level is the shared instance used by
    mao.core.llm across all agents.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Per-model counters: {model: {"input": int, "output": int, "requests": int}}
        self._per_model: dict[str, dict[str, int]] = {}
        # Most-recent rate-limit headers from Groq response
        self._last_rate_limit_info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Record a completed LLM call.

        Args:
            model:         Groq model name (e.g. "llama-3.1-8b-instant").
            input_tokens:  Prompt token count from usage.prompt_tokens.
            output_tokens: Completion token count from usage.completion_tokens.
            headers:       Optional dict of Groq response headers for rate-limit
                           fields (x-ratelimit-limit-requests, etc.).
        """
        with self._lock:
            if model not in self._per_model:
                self._per_model[model] = {"input": 0, "output": 0, "requests": 0}
            bucket = self._per_model[model]
            bucket["input"]    += max(0, input_tokens)
            bucket["output"]   += max(0, output_tokens)
            bucket["requests"] += 1

            if headers:
                self._last_rate_limit_info = _extract_rate_limit(headers)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return aggregated usage statistics and estimated cost.

        Returns a dict with keys:
          total_input_tokens  — int
          total_output_tokens — int
          total_tokens        — int
          estimated_cost_usd  — float, rounded to 6 decimal places
          per_model           — {model: {input, output, requests, cost_usd}}
          rate_limit_info     — last observed rate-limit header values (may be {})
        """
        with self._lock:
            total_input  = 0
            total_output = 0
            total_cost   = 0.0
            per_model_out: dict[str, dict[str, Any]] = {}

            for model, counts in self._per_model.items():
                pricing = _PRICING.get(model, _PRICING["default"])
                cost = (
                    counts["input"]  / 1_000_000 * pricing["input"]
                    + counts["output"] / 1_000_000 * pricing["output"]
                )
                total_input  += counts["input"]
                total_output += counts["output"]
                total_cost   += cost
                per_model_out[model] = {
                    "input_tokens":  counts["input"],
                    "output_tokens": counts["output"],
                    "requests":      counts["requests"],
                    "cost_usd":      round(cost, 6),
                }

            return {
                "total_input_tokens":  total_input,
                "total_output_tokens": total_output,
                "total_tokens":        total_input + total_output,
                "estimated_cost_usd":  round(total_cost, 6),
                "per_model":           per_model_out,
                "rate_limit_info":     dict(self._last_rate_limit_info),
            }

    def reset(self) -> None:
        """Clear all accumulated counters (e.g. at session boundary)."""
        with self._lock:
            self._per_model.clear()
            self._last_rate_limit_info.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_rate_limit(headers: dict[str, str]) -> dict[str, Any]:
    """Parse Groq rate-limit headers into a typed dict.

    Groq header names (case-insensitive):
      x-ratelimit-limit-requests
      x-ratelimit-limit-tokens
      x-ratelimit-remaining-requests
      x-ratelimit-remaining-tokens
      x-ratelimit-reset-requests
      x-ratelimit-reset-tokens
    """
    result: dict[str, Any] = {}
    lower_headers = {k.lower(): v for k, v in headers.items()}

    def _int(key: str) -> int | None:
        val = lower_headers.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _str(key: str) -> str | None:
        return lower_headers.get(key)

    limit_req  = _int("x-ratelimit-limit-requests")
    limit_tok  = _int("x-ratelimit-limit-tokens")
    rem_req    = _int("x-ratelimit-remaining-requests")
    rem_tok    = _int("x-ratelimit-remaining-tokens")

    if limit_req  is not None: result["limit_requests"]     = limit_req
    if limit_tok  is not None: result["limit_tokens"]        = limit_tok
    if rem_req    is not None: result["remaining_requests"]  = rem_req
    if rem_tok    is not None: result["remaining_tokens"]    = rem_tok

    reset_req = _str("x-ratelimit-reset-requests")
    reset_tok = _str("x-ratelimit-reset-tokens")
    if reset_req: result["reset_requests"] = reset_req
    if reset_tok: result["reset_tokens"]   = reset_tok

    return result


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
tracker = GroqUsageTracker()


# ---------------------------------------------------------------------------
# Usage example / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tracker.record("llama-3.1-8b-instant", input_tokens=500, output_tokens=200)
    tracker.record("llama-3.3-70b-versatile", input_tokens=1000, output_tokens=400)
    tracker.record(
        "llama-3.1-8b-instant",
        input_tokens=300,
        output_tokens=100,
        headers={
            "x-ratelimit-limit-requests": "30",
            "x-ratelimit-remaining-requests": "28",
            "x-ratelimit-limit-tokens": "6000",
            "x-ratelimit-remaining-tokens": "5200",
        },
    )

    import json
    print(json.dumps(tracker.get_summary(), indent=2))
