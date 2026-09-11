"""Splits genuinely multi-part questions into independent sub-questions.

P2-15: decomposition is *conditional*. Most biomedical queries are single-part,
and an unconditional decomposer spent one extraction-model call on every single
request for a list of length one. A cheap deterministic pre-check
(:func:`needs_decomposition`) now gates the model call, so a single-part query
costs nothing.
"""
from __future__ import annotations

import json
import logging
import re
import time

from mao.prompts import get_prompt
from mao.providers import gateway
from mao.trust.classes import TrustClass
from mao.trust.egress.policy import EgressPurpose
from mao.providers.registry import ModelRole

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic multi-part pre-check (no model call)
# ---------------------------------------------------------------------------

_INTERROGATIVE = (
    r"what|how|why|when|where|which|who|whom|whose|"
    r"does|do|did|can|could|should|would|will"
)

# Imperatives that introduce a second request: "…and EXPLAIN the protocol",
# "…; INCLUDE the contraindications". Real multi-part clinical questions use
# these far more often than a second interrogative.
_IMPERATIVE = (
    r"list|explain|describe|include|give|tell|summari[sz]e|provide|outline|"
    r"detail|compare|discuss|state|name|identify"
)

_CONNECTOR = r"(?:\band\b|\bor\b|\balso\b|\bplus\b|\bas well as\b|\balong with\b|;)"

# A connector followed by a second interrogative OR a second imperative.
#   "What is amyloid AND HOW does tau ..."      (interrogative)
#   "List the treatments AND THEIR side effects" (possessive continuation)
#   "Describe lecanemab AND EXPLAIN the protocol" (imperative)
_CONNECTED_QUESTION = re.compile(
    rf"{_CONNECTOR}\s+(?:{_INTERROGATIVE}|{_IMPERATIVE}|their|its|the\s+recommended)\b",
    re.IGNORECASE,
)

# "along with" / "as well as" introduce a second subject regardless of what
# follows, so they are sufficient on their own.
_ADDITIVE = re.compile(r"\balong with\b|\bas well as\b", re.IGNORECASE)

# A serial list before a connector: "aducanumab, memantine, and donepezil".
_SERIAL_LIST = re.compile(r",\s*\w[\w\-]*\s*,?\s+(?:and|or)\b", re.IGNORECASE)

# A choice between two named options: "memantine OR donepezil".
_ALTERNATIVE = re.compile(r"\b\w[\w\-]{3,}\s+or\s+\w[\w\-]{3,}\b", re.IGNORECASE)

# A new sentence that opens with an imperative is a second request:
#   "What is the thrombolysis window? INCLUDE the contraindications."
_SECOND_SENTENCE_IMPERATIVE = re.compile(
    rf"[.?!]\s+(?:{_IMPERATIVE})\b", re.IGNORECASE
)

# Fixed collocations that are one concept despite the connector. Without this
# exclusion the coordinated-noun-phrase rule below would decompose them.
_FIXED_PAIRS = re.compile(
    r"\b(?:signs and symptoms|safety and efficacy|risks? and benefits?|"
    r"diagnosis and treatment|health and social care|terms and conditions|"
    r"morbidity and mortality)\b",
    re.IGNORECASE,
)

# A connector joining two multi-word noun phrases, each naming its own subject:
#   "vascular dementia AND LEWY BODY dementia"
#   "amyloid PET imaging AND TAU PET imaging"
_COORDINATED_NOUN_PHRASE = re.compile(
    r"\b(?:and|or)\s+(?:[a-z][\w\-]*\s+){1,3}[a-z][\w\-]*\b", re.IGNORECASE
)

# Narrative clinical history, not a second request: "diagnosed with MCI AND IS
# now on donepezil". These read as multi-part to a naive connector rule.
_NARRATIVE_CONJUNCTION = re.compile(
    r"\b(?:and|or)\s+(?:is|are|was|were|has|have|had|then|now|subsequently)\b",
    re.IGNORECASE,
)

# Explicit comparison requests always have at least two retrievable subjects.
_COMPARISON = re.compile(
    r"\bcompare\b|\bdifference(?:s)? between\b|\bversus\b|\bvs\.?\b|\bcontrast\b",
    re.IGNORECASE,
)


def needs_decomposition(query: str) -> bool:
    """True when *query* is genuinely multi-part and worth a model call.

    A false negative is more expensive than it first appears. Besides costing a
    single coarse retrieval pass, it collapses ``sub_queries`` to ``[query]``,
    and `senior_supervisor_node` then scores completeness against that one
    question — so an answer covering only half of a two-part clinical question
    is still marked complete. The rule therefore leans towards decomposing.
    """
    text = (query or "").strip()
    if not text:
        return False

    # Narrative history ("diagnosed with MCI and is now on donepezil") reads as
    # a conjunction but asks one question. Checked first so it can veto.
    narrative = _NARRATIVE_CONJUNCTION.search(text)

    if text.count("?") > 1:
        return True
    if _COMPARISON.search(text):
        return True
    if _ADDITIVE.search(text):
        return True
    if _SECOND_SENTENCE_IMPERATIVE.search(text):
        return True
    if narrative:
        return False
    if _SERIAL_LIST.search(text):
        return True
    if _CONNECTED_QUESTION.search(text):
        return True
    if _ALTERNATIVE.search(text):
        return True
    if _FIXED_PAIRS.search(text):
        return False
    return bool(_COORDINATED_NOUN_PHRASE.search(text))


# ---------------------------------------------------------------------------
# Model-backed decomposition
# ---------------------------------------------------------------------------

def _chat_with_retry(messages: list[dict], *, max_retries: int = 2, **kwargs) -> str:
    """Retry wrapper around the model gateway.

    Retries up to *max_retries* times with linear back-off (0.5 s, 1.0 s)
    before re-raising the final exception.  Keeps the decomposer resilient to
    transient Ollama / Groq timeouts without hiding hard failures.
    """
    for attempt in range(max_retries + 1):
        try:
            return gateway.complete(
                messages=messages,
                purpose=EgressPurpose.ROUTING,
                trust_class=TrustClass.SAFE_DERIVED_TEXT,
                **kwargs,
            ).text
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            wait = 0.5 * (attempt + 1)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries + 1,
                exc,
                wait,
            )
            time.sleep(wait)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap JSON in."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        stripped = "\n".join(inner).strip()
    return stripped


def _parse_sub_queries(payload: object) -> list[str] | None:
    """Accept the registered ``{"sub_queries": [...]}`` contract, or a bare list."""
    if isinstance(payload, dict):
        payload = payload.get("sub_queries")
    if isinstance(payload, list) and all(isinstance(p, str) for p in payload):
        cleaned = [p.strip() for p in payload if p.strip()]
        return cleaned or None
    return None


def _llm_decompose(query: str) -> list[str]:
    spec = get_prompt("decomposer.split")
    raw = _chat_with_retry(
        messages=[
            {"role": "system", "content": spec.template},
            {"role": "user", "content": query},
        ],
        role=ModelRole.EXTRACTION_FAST,
        max_tokens=300,
        temperature=0.0,
    ).strip()
    try:
        parsed = _parse_sub_queries(json.loads(_strip_fences(raw)))
    except json.JSONDecodeError:
        logger.warning("Decomposer returned non-JSON: %s", raw)
        return [query]
    return parsed or [query]


def decompose_query(query: str) -> list[str]:
    """Return the sub-questions of *query*, spending a model call only if needed."""
    if not needs_decomposition(query):
        logger.debug("Decomposer: single-part query, no model call")
        return [query]
    return _llm_decompose(query)


def decomposer_node(state: dict) -> dict:
    """Decompose the query the protected boundary already produced.

    This used to call `scrub_pii` again. `state["user_query"]` is the boundary's
    output, so the second pass was a second semantic transformation of an
    already-transformed string, and it over-redacted a clinical line the first
    pass had correctly left alone:

        'Patient Name:\\nMRN:\\nHarold Nkemdirim\\nRockwood Frailty\\n'
          x1 -> 'Patient Name:\\nMRN:\\n[NAME]\\nRockwood Frailty\\n'
          x2 -> 'Patient Name:\\nMRN:\\n[NAME]\\n[NAME]\\n'

    Two mechanisms were tried to make the second pass a no-op, and both had to
    recognise the first pass's output from a marker or a position in text the
    caller controls — so both were forgeable, and removing the forgeable one
    reopened the destruction. The property is not achievable while a second pass
    exists, so there is no second pass. See `mao/trust/inputs/boundary.py`.
    """
    clean_query = state.get("user_query", "")
    sub_queries = decompose_query(clean_query)
    return {
        **state,
        "pii_scrubbed_query": clean_query,
        "sub_queries": sub_queries,
        "missing_sub_queries": list(sub_queries),
    }
