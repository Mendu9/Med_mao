"""Domain supervisor — flags claims the retrieved sources do not support.

P0-2  This node used to overwrite `state["response"]` with an LLM-generated
      `grounded_summary`. Two things went wrong at once: the mandatory clinical
      disclaimer appended by `clinical_agent` was discarded, and a response that
      had been through the pipeline was replaced by one that had not. The
      supervisor now observes and reports; it never edits the answer. Prompt
      `domain_supervisor.reconcile` v2 no longer even asks for a rewrite.

P0-1  It also skipped itself whenever `_want_stream` was set, so a streamed
      answer was never checked for grounding. Transport is not a safety input.
"""
from __future__ import annotations

import json
import logging
import re

from mao.prompts import get_prompt
from mao.providers import gateway
from mao.providers.registry import ModelRole
from mao.schemas.evidence import as_text

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

logger = logging.getLogger(__name__)

_PROMPT_NAME = "domain_supervisor.reconcile"
_MAX_TOKENS = 500

# Advisory, so deliberately NOT ModelRole.SAFETY_JUDGE.
#
# This node observes grounding and blocks nothing — P0-2 removed its ability to
# edit the response, and nothing reads `ungrounded_claims` to gate anything.
#
# Running it on the safety model meant a third of that model's per-request token
# spend went on a signal no code consumed, against a per-model quota the council
# and the judge need. Provider limits are per model, so moving an advisory check
# to a different model is not a smaller safety budget; it is the same safety
# budget, no longer shared with work that cannot affect a safety decision.
# Structured extraction from text is what EXTRACTION_FAST is for.
_ROLE = ModelRole.EXTRACTION_FAST


def _parse_ungrounded_claims(raw: str) -> list[str]:
    """Pull `ungrounded_claims` out of the model's JSON. Never raises."""
    try:
        cleaned = _JSON_FENCE_RE.sub("", (raw or "").strip())
        # Strip to the outermost braces in case of leading/trailing prose.
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start == -1 or brace_end <= brace_start:
            return []
        parsed = json.loads(cleaned[brace_start : brace_end + 1])
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    claims = parsed.get("ungrounded_claims", [])
    if not isinstance(claims, list):
        return []
    return [str(c) for c in claims]


def domain_supervisor_node(state: dict) -> dict:
    answer = state.get("response", state.get("answer", "")) or ""

    # Nothing to reconcile. Judged on its own merits — deliberately NOT
    # conditioned on `_want_stream` (P0-1).
    if not answer.strip():
        return {**state, "ungrounded_claims": []}

    rag_chunks = state.get("retrieved_docs", []) or []
    web_results = state.get("web_results", []) or []

    spec = get_prompt(_PROMPT_NAME)
    # `as_text`, not `str(c)` — see mao/schemas/evidence.py. Asking a model
    # which claims are ungrounded, while showing it dict reprs as the ground,
    # produced ungrounded-claim lists that described the punctuation.
    context = "\n".join([
        "RAG CHUNKS:", *[as_text(c) for c in rag_chunks[:4]],
        "WEB RESULTS:", *[as_text(w) for w in web_results[:3]],
        "DRAFT ANSWER:", answer,
    ])

    try:
        completion = gateway.complete(
            role=_ROLE,
            messages=[
                {"role": "system", "content": spec.template},
                {"role": "user", "content": context},
            ],
            temperature=0.0,
            max_tokens=_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 - grounding is advisory; never fail the request
        logger.error("Domain supervisor call failed: %s", exc)
        return {**state, "ungrounded_claims": []}

    # NOTE: `state["response"]` is deliberately not assigned anywhere in this
    # function. Reintroducing it reintroduces P0-2.
    return {
        **state,
        "ungrounded_claims": _parse_ungrounded_claims(completion.text),
        "grounding_prompt_ref": spec.trace_ref,
    }
