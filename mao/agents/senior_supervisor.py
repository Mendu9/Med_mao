"""Senior supervisor — checks each decomposed sub-question against the answer.

P0-1  This node used to skip its check whenever `_want_stream` was set and the
      answer was still empty, which made supervision a function of the transport.
      The only legitimate reasons to skip the model call are having no
      sub-questions, or having no answer — and an empty answer answers none of
      its sub-questions, a conclusion reachable without a 70B call.
"""
from __future__ import annotations

import json
import logging
import re

from mao.prompts import get_prompt
from mao.providers import gateway
from mao.providers.registry import ModelRole

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

logger = logging.getLogger(__name__)

_PROMPT_NAME = "senior_supervisor.completeness"
_MAX_TOKENS = 300

# Advisory, so deliberately NOT ModelRole.SAFETY_JUDGE — see the note in
# domain_supervisor.py. Nothing reads `completeness_ok` to gate anything, and
# provider quotas are per model, so this call was spending the safety model's
# budget on a signal that cannot affect a safety decision.
_ROLE = ModelRole.EXTRACTION_FAST


def _parse_missing(raw: str) -> list[str]:
    """Pull `missing` out of the model's JSON. Never raises."""
    try:
        cleaned = _JSON_FENCE_RE.sub("", (raw or "").strip())
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start == -1 or brace_end <= brace_start:
            return []
        parsed = json.loads(cleaned[brace_start : brace_end + 1])
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    missing = parsed.get("missing", [])
    if not isinstance(missing, list):
        return []
    return [str(m) for m in missing]


def senior_supervisor_node(state: dict) -> dict:
    sub_queries = state.get("sub_queries", []) or []
    answer = state.get("response", state.get("answer", "")) or ""

    if not sub_queries:
        return {**state, "completeness_ok": True, "missing_sub_queries": []}

    # An empty answer answers nothing. Deliberately NOT conditioned on
    # `_want_stream`: reporting "complete" here because the request happened to
    # be streamed is exactly the defect (P0-1).
    if not answer.strip():
        return {
            **state,
            "completeness_ok": False,
            "missing_sub_queries": list(sub_queries),
        }

    spec = get_prompt(_PROMPT_NAME)
    prompt = f"SUB-QUESTIONS:\n{json.dumps(sub_queries)}\n\nANSWER:\n{answer}"

    try:
        completion = gateway.complete(
            role=_ROLE,
            messages=[
                {"role": "system", "content": spec.template},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 - completeness is advisory
        logger.error("Senior supervisor call failed: %s", exc)
        return {**state, "completeness_ok": False, "missing_sub_queries": []}

    missing = _parse_missing(completion.text)
    return {
        **state,
        "completeness_ok": len(missing) == 0,
        "missing_sub_queries": missing,
        "completeness_prompt_ref": spec.trace_ref,
    }
