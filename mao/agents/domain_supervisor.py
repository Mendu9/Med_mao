from __future__ import annotations

import json
import logging
import re
from mao.core.config import CLINICAL_MODEL

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

logger = logging.getLogger(__name__)

_SYSTEM = """You are a Domain Supervisor. Given retrieved RAG chunks and web search results,
you must reconcile them and check source grounding.
Output JSON with keys: "grounded_summary" (str), "ungrounded_claims" (list[str]), "sources_used" (list[str])"""

def domain_supervisor_node(state: dict) -> dict:
    from mao.core.llm import chat
    answer = state.get("response", state.get("answer", ""))

    # Skip supervision when streaming path deferred the LLM call — response is empty
    if state.get("_want_stream") and not answer:
        return {**state, "ungrounded_claims": []}

    rag_chunks = state.get("retrieved_docs", [])
    web_results = state.get("web_results", [])

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
        cleaned = _JSON_FENCE_RE.sub("", raw.strip())
        # Also strip to the outermost braces in case of leading/trailing text
        brace_start = cleaned.find("{")
        brace_end   = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]
        parsed = json.loads(cleaned)
        grounded = parsed.get("grounded_summary") or answer
        ungrounded = parsed.get("ungrounded_claims", [])
    except json.JSONDecodeError:
        grounded = answer
        ungrounded = []

    return {**state, "response": grounded, "ungrounded_claims": ungrounded}
