import json
import logging
from mao.core.config import FAST_MODEL
from mao.core.pii_scrubber import scrub_pii

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a medical query parser. Given a user query, decompose it into a JSON list "
    "of self-contained sub-questions. If the query is already a single question, return a "
    "list with one item. Output ONLY valid JSON, no prose. Example: "
    '[\"What is amyloid?\", \"How does tau cause neurodegeneration?\"]'
)

def _llm_decompose(query: str) -> list[str]:
    from mao.core.llm import chat
    raw = chat(
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": query}],
        model=FAST_MODEL,
        max_tokens=300,
        temperature=0.0,
    ).strip()
    try:
        parts = json.loads(raw)
        if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
            return [p.strip() for p in parts if p.strip()]
    except json.JSONDecodeError:
        logger.warning("Decomposer returned non-JSON: %s", raw)
    return [query]

def decompose_query(query: str) -> list[str]:
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
