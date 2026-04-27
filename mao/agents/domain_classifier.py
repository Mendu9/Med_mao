import logging
from mao.core.config import FAST_MODEL

logger = logging.getLogger(__name__)

_VALID_DOMAINS = {"alzheimer", "stroke", "general"}

_SYSTEM = (
    "You are a medical domain classifier. Given a clinical query, respond with EXACTLY "
    "one word from: alzheimer, stroke, general. No explanation."
)

def _llm_classify(query: str) -> str:
    from mao.core.llm import chat
    return chat(
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": query}],
        model=FAST_MODEL,
        max_tokens=5,
        temperature=0.0,
    ).strip().lower()

def classify_domain(query: str) -> str:
    label = _llm_classify(query)
    return label if label in _VALID_DOMAINS else "general"

def classifier_node(state: dict) -> dict:
    query = state.get("pii_scrubbed_query") or state.get("query", "")
    domain = classify_domain(query)
    return {**state, "domain": domain}
