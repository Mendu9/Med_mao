"""Assigns a biomedical domain, used downstream to scope retrieval."""
from __future__ import annotations

import logging

from mao.prompts import get_prompt
from mao.providers.gateway import model_id_for
from mao.providers.registry import ModelRole

logger = logging.getLogger(__name__)

_VALID_DOMAINS = {"alzheimer", "stroke", "general"}


def _llm_classify(query: str) -> str:
    from mao.core.llm import chat

    return chat(
        messages=[
            {"role": "system", "content": get_prompt("domain.classify").template},
            {"role": "user", "content": query},
        ],
        model=model_id_for(ModelRole.EXTRACTION_FAST),
        max_tokens=5,
        temperature=0.0,
    ).strip().lower()


def classify_domain(query: str) -> str:
    label = _llm_classify(query)
    return label if label in _VALID_DOMAINS else "general"


def classifier_node(state: dict) -> dict:
    query = state.get("pii_scrubbed_query") or state.get("user_query", "")
    domain = classify_domain(query)
    return {**state, "domain": domain}
