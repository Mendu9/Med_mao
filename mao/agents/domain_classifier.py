"""Assigns a biomedical domain, used downstream to scope retrieval."""
from __future__ import annotations

import logging

from mao.prompts import get_prompt
from mao.providers import gateway
from mao.providers.registry import ModelRole

logger = logging.getLogger(__name__)

_VALID_DOMAINS = {"alzheimer", "stroke", "general"}


def _llm_classify(query: str) -> str:
    return gateway.complete(
        role=ModelRole.EXTRACTION_FAST,
        messages=[
            {"role": "system", "content": get_prompt("domain.classify").template},
            {"role": "user", "content": query},
        ],
        # Not 5: a model that emits any preamble returns an empty string at that
        # budget, and every query then silently classifies as "general".
        max_tokens=32,
        temperature=0.0,
    ).text.strip().lower()


def classify_domain(query: str) -> str:
    label = _llm_classify(query)
    return label if label in _VALID_DOMAINS else "general"


def classifier_node(state: dict) -> dict:
    query = state.get("pii_scrubbed_query") or state.get("user_query", "")
    domain = classify_domain(query)
    return {**state, "domain": domain}
