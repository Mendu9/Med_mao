import logging
from mao.core.config import FAST_MODEL
from mao.core.retry import with_groq_retry

logger = logging.getLogger(__name__)

_VALID_DOMAINS = {"alzheimer", "stroke", "general"}

_SYSTEM = (
    "You are a medical domain classifier. Given a clinical query, respond with EXACTLY "
    "one word from: alzheimer, stroke, general. No explanation."
)

def _llm_classify(query: str) -> str:
    from mao.core.llm import get_client
    client = get_client()
    resp = with_groq_retry(lambda: client.chat.completions.create(
        model=FAST_MODEL,
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": query}],
        max_tokens=5,
        temperature=0.0,
    ))
    return resp.choices[0].message.content.strip().lower()

def classify_domain(query: str) -> str:
    label = _llm_classify(query)
    return label if label in _VALID_DOMAINS else "general"

def classifier_node(state: dict) -> dict:
    query = state.get("pii_scrubbed_query") or state.get("query", "")
    domain = classify_domain(query)
    return {**state, "domain": domain}
