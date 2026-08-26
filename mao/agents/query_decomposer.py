"""Splits genuinely multi-part questions into independent sub-questions.

P2-15: decomposition is *conditional*. Most biomedical queries are single-part,
and an unconditional decomposer spent one extraction-model call on every single
request for a list of length one. A cheap deterministic pre-check
(:func:`needs_decomposition`) now gates the model call, so a single-part query
costs nothing.
"""
from __future__ import annotations

import json
import logging
import re
import time

from mao.core.pii_scrubber import scrub_pii
from mao.prompts import get_prompt
from mao.providers.gateway import model_id_for
from mao.providers.registry import ModelRole

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic multi-part pre-check (no model call)
# ---------------------------------------------------------------------------

_INTERROGATIVE = (
    r"what|how|why|when|where|which|who|whom|whose|"
    r"does|do|did|is|are|was|were|can|could|should|would|will"
)

# A connector immediately followed by a second interrogative:
#   "What is amyloid AND HOW does tau ...", "Explain APOE4; ALSO WHAT ..."
_CONNECTED_QUESTION = re.compile(
    rf"(?:\band\b|\bor\b|\balso\b|\bplus\b|\bas well as\b|;)\s+(?:{_INTERROGATIVE})\b",
    re.IGNORECASE,
)

# Explicit comparison requests always have at least two retrievable subjects.
_COMPARISON = re.compile(
    r"\bcompare\b|\bdifference(?:s)? between\b|\bversus\b|\bvs\.?\b|\bcontrast\b",
    re.IGNORECASE,
)


def needs_decomposition(query: str) -> bool:
    """True when *query* is genuinely multi-part and worth a model call.

    Deliberately conservative: a false negative costs one extra retrieval pass
    over the whole query, while a false positive costs a model call on every
    such request.
    """
    text = (query or "").strip()
    if not text:
        return False
    if text.count("?") > 1:
        return True
    if _COMPARISON.search(text):
        return True
    return bool(_CONNECTED_QUESTION.search(text))


# ---------------------------------------------------------------------------
# Model-backed decomposition
# ---------------------------------------------------------------------------

def _chat_with_retry(messages: list[dict], *, max_retries: int = 2, **kwargs) -> str:
    """Retry wrapper around ``chat()``.

    Retries up to *max_retries* times with linear back-off (0.5 s, 1.0 s)
    before re-raising the final exception.  Keeps the decomposer resilient to
    transient Ollama / Groq timeouts without hiding hard failures.
    """
    from mao.core.llm import chat  # local import avoids circular dependency at module load

    for attempt in range(max_retries + 1):
        try:
            return chat(messages=messages, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            wait = 0.5 * (attempt + 1)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries + 1,
                exc,
                wait,
            )
            time.sleep(wait)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap JSON in."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        stripped = "\n".join(inner).strip()
    return stripped


def _parse_sub_queries(payload: object) -> list[str] | None:
    """Accept the registered ``{"sub_queries": [...]}`` contract, or a bare list."""
    if isinstance(payload, dict):
        payload = payload.get("sub_queries")
    if isinstance(payload, list) and all(isinstance(p, str) for p in payload):
        cleaned = [p.strip() for p in payload if p.strip()]
        return cleaned or None
    return None


def _llm_decompose(query: str) -> list[str]:
    spec = get_prompt("decomposer.split")
    raw = _chat_with_retry(
        messages=[
            {"role": "system", "content": spec.template},
            {"role": "user", "content": query},
        ],
        model=model_id_for(ModelRole.EXTRACTION_FAST),
        max_tokens=300,
        temperature=0.0,
    ).strip()
    try:
        parsed = _parse_sub_queries(json.loads(_strip_fences(raw)))
    except json.JSONDecodeError:
        logger.warning("Decomposer returned non-JSON: %s", raw)
        return [query]
    return parsed or [query]


def decompose_query(query: str) -> list[str]:
    """Return the sub-questions of *query*, spending a model call only if needed."""
    if not needs_decomposition(query):
        logger.debug("Decomposer: single-part query, no model call")
        return [query]
    return _llm_decompose(query)


def decomposer_node(state: dict) -> dict:
    clean_query = scrub_pii(state.get("user_query", ""))
    sub_queries = decompose_query(clean_query)
    return {
        **state,
        "pii_scrubbed_query": clean_query,
        "sub_queries": sub_queries,
        "missing_sub_queries": list(sub_queries),
    }
