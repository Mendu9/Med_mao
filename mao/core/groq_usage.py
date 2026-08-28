"""
mao/core/groq_usage.py
-----------------------
In-memory Groq API token usage tracker.
Accumulates input/output token counts per model across the process lifetime.
Exposes a module-level singleton `tracker` used by llm.py and main.py.

P1-18: pricing is read from the ModelRegistry's `cost_per_1m_input_usd` /
`cost_per_1m_output_usd`, not from a second literal table. The old table had
drifted — it still priced `llama-3.1-70b-versatile` and `mixtral-8x7b-32768`,
both of which the registry records as RETIRED — so every cost figure derived
from it was wrong for those ids.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Used only when the registry has no usable price for an id: an operator-supplied
# model, or a retired record kept for refusal purposes (which carries no price).
# Deliberately pessimistic so unknown spend is over- rather than under-reported.
_FALLBACK_INPUT_PER_1M = 0.50
_FALLBACK_OUTPUT_PER_1M = 0.80

_warned_unpriced: set[str] = set()


@dataclass(frozen=True)
class ModelPrice:
    """Resolved price for one model id, plus where the numbers came from."""

    input_per_1m: float
    output_per_1m: float
    source: str  # "registry" | "unknown"

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_1m + output_tokens * self.output_per_1m
        ) / 1_000_000


def price_for(model_id: str) -> ModelPrice:
    """Price a model id from the ModelRegistry, degrading gracefully.

    Imported lazily: `mao.core.llm` imports this module, and the provider
    gateway imports back into core, so a module-level import would risk a cycle.
    """
    record = None
    try:
        from mao.providers.gateway import registry

        record = registry().get(model_id)
    except Exception as exc:  # pragma: no cover - registry construction is total
        logger.debug("Model registry unavailable for pricing %r: %s", model_id, exc)

    if record is not None:
        input_price = float(record.cost_per_1m_input_usd)
        output_price = float(record.cost_per_1m_output_usd)
        if input_price > 0.0 or output_price > 0.0:
            return ModelPrice(input_price, output_price, "registry")

    if model_id not in _warned_unpriced:
        _warned_unpriced.add(model_id)
        logger.warning(
            "No registry pricing for model %r — estimating at $%.2f/$%.2f per 1M tokens",
            model_id,
            _FALLBACK_INPUT_PER_1M,
            _FALLBACK_OUTPUT_PER_1M,
        )
    return ModelPrice(_FALLBACK_INPUT_PER_1M, _FALLBACK_OUTPUT_PER_1M, "unknown")


class _UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._per_model: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0, "cost_usd": 0.0}
        )

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        cost = price_for(model).cost_for(input_tokens, output_tokens)
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
