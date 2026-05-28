"""
mao/core/groq_usage.py
-----------------------
In-memory Groq API token usage tracker.
Accumulates input/output token counts per model across the process lifetime.
Exposes a module-level singleton `tracker` used by llm.py and main.py.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

# Groq pricing (USD per 1M tokens)
_PRICE_PER_1M: dict[str, dict[str, float]] = {
    "llama-3.1-8b-instant":     {"input": 0.05,  "output": 0.08},
    "llama-3.3-70b-versatile":  {"input": 0.59,  "output": 0.79},
    "llama-3.1-70b-versatile":  {"input": 0.59,  "output": 0.79},
    "mixtral-8x7b-32768":       {"input": 0.24,  "output": 0.24},
    "gemma2-9b-it":             {"input": 0.20,  "output": 0.20},
}
_DEFAULT_PRICE = {"input": 0.50, "output": 0.80}


class _UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._per_model: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0, "cost_usd": 0.0}
        )

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        price = _PRICE_PER_1M.get(model, _DEFAULT_PRICE)
        cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
        with self._lock:
            bucket = self._per_model[model]
            bucket["input_tokens"]  += input_tokens
            bucket["output_tokens"] += output_tokens
            bucket["requests"]      += 1
            bucket["cost_usd"]      += cost

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            total_input  = sum(v["input_tokens"]  for v in self._per_model.values())
            total_output = sum(v["output_tokens"]  for v in self._per_model.values())
            total_cost   = sum(v["cost_usd"]       for v in self._per_model.values())
            return {
                "total_tokens":       total_input + total_output,
                "input_tokens":       total_input,
                "output_tokens":      total_output,
                "estimated_cost_usd": round(total_cost, 6),
                "per_model":          dict(self._per_model),
            }

    def reset(self) -> None:
        with self._lock:
            self._per_model.clear()


tracker = _UsageTracker()
