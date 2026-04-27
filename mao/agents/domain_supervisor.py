import json
import logging
from mao.core.config import CLINICAL_MODEL

logger = logging.getLogger(__name__)

_SYSTEM = """You are a Domain Supervisor. Given retrieved RAG chunks and web search results,
you must reconcile them and check source grounding.
Output JSON with keys: "grounded_summary" (str), "ungrounded_claims" (list[str]), "sources_used" (list[str])"""

def domain_supervisor_node(state: dict) -> dict:
    from mao.core.llm import chat
    rag_chunks = state.get("retrieved_docs", [])
    web_results = state.get("web_results", [])
    answer = state.get("response", state.get("answer", ""))

    context = "\n".join([
        "RAG CHUNKS:", *[str(c) for c in rag_chunks[:4]],
        "WEB RESULTS:", *[str(w) for w in web_results[:3]],
        "DRAFT ANSWER:", answer,
    ])

    raw = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": context},
        ],
        model=CLINICAL_MODEL,
        max_tokens=500,
        temperature=0.0,
    )

    try:
        parsed = json.loads(raw)
        grounded = parsed.get("grounded_summary") or answer
        ungrounded = parsed.get("ungrounded_claims", [])
    except json.JSONDecodeError:
        grounded = answer
        ungrounded = []

    return {**state, "response": grounded, "ungrounded_claims": ungrounded}
