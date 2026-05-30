from __future__ import annotations

import json
import logging
from mao.core.config import CLINICAL_MODEL

logger = logging.getLogger(__name__)

_SYSTEM = """You are the Senior Clinical Supervisor. You receive a list of sub-questions and the current answer.
For each sub-question, determine if it is answered.
Output JSON only: {"answered": ["sub-q 1", ...], "missing": ["sub-q 2", ...]}"""

def senior_supervisor_node(state: dict) -> dict:
    sub_queries = state.get("sub_queries", [])
    answer = state.get("response", state.get("answer", ""))

    if not sub_queries:
        return {**state, "completeness_ok": True, "missing_sub_queries": []}

    from mao.core.llm import chat
    prompt = f"SUB-QUESTIONS:\n{json.dumps(sub_queries)}\n\nANSWER:\n{answer}"
    raw = chat(
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        model=CLINICAL_MODEL,
        max_tokens=300,
        temperature=0.0,
    ).strip()
    try:
        parsed = json.loads(raw)
        missing = parsed.get("missing", [])
    except json.JSONDecodeError:
        missing = []

    return {**state, "completeness_ok": len(missing) == 0, "missing_sub_queries": missing}
