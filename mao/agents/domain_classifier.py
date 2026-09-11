"""Assigns a biomedical domain, used downstream to scope retrieval."""
from __future__ import annotations

import logging

from mao.prompts import get_prompt
from mao.providers import gateway
from mao.trust.classes import TrustClass
from mao.trust.egress.policy import EgressPurpose
from mao.providers.registry import ModelRole

logger = logging.getLogger(__name__)

_VALID_DOMAINS = {"alzheimer", "stroke", "general"}

# Budget for the one-word domain label. Named rather than inline so the live
# call-site probe can assert against the value this module actually uses — an
# `inspect.getsource` check for the string "max_tokens=5" passed at 6 and could
# not see the bound model at all.
#
# Not 5: a model that emits any preamble returns an empty string at that budget,
# and every query then silently classifies as "general". The model's analysis
# channel is paid for separately by the gateway, from the record's declared
# reasoning overhead.
_DOMAIN_MAX_TOKENS = 32


def _llm_classify(query: str) -> str:
    return gateway.complete(
        role=ModelRole.EXTRACTION_FAST,
        messages=[
            {"role": "system", "content": get_prompt("domain.classify").template},
            {"role": "user", "content": query},
        ],
        purpose=EgressPurpose.ROUTING,
        trust_class=TrustClass.SAFE_DERIVED_TEXT,
        max_tokens=_DOMAIN_MAX_TOKENS,
        temperature=0.0,
    ).text.strip().lower()


def classify_domain(query: str) -> str:
    label = _llm_classify(query)
    return label if label in _VALID_DOMAINS else "general"


def classifier_node(state: dict) -> dict:
    query = state.get("pii_scrubbed_query") or state.get("user_query", "")
    domain = classify_domain(query)
    return {**state, "domain": domain}
