from __future__ import annotations

import logging
from mao.core.config import NLI_MODEL, NLI_ENTAILMENT_THRESHOLD

logger = logging.getLogger(__name__)

_cross_encoder = None
_nli_disabled = False  # latched True on load failure to avoid repeated crashes

_MAX_PREMISE_CHARS = 2000  # ~512 tokens; truncate before sending to cross-encoder


def _get_encoder():
    global _cross_encoder, _nli_disabled
    if _nli_disabled:
        return None
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(NLI_MODEL)
        except Exception as exc:
            _nli_disabled = True
            logger.warning("NLI model failed to load — disabling NLI checks: %s", exc)
            return None
    return _cross_encoder


def _safe_result(claim: str) -> dict:
    return {"claim": claim, "entailed": False, "score": 0.0, "contradiction_score": 0.0}


def check_claim(premise: str, claim: str) -> dict:
    enc = _get_encoder()
    if enc is None:
        return _safe_result(claim)
    if len(premise) > _MAX_PREMISE_CHARS:
        logger.warning("NLI premise truncated from %d to %d chars", len(premise), _MAX_PREMISE_CHARS)
        premise = premise[:_MAX_PREMISE_CHARS]
    scores = enc.predict([[premise, claim]])[0]
    if len(scores) < 3:
        logger.warning("NLI model returned unexpected score shape: %s", scores)
        return _safe_result(claim)
    entailment_score = float(scores[2])
    return {
        "claim": claim,
        "entailed": entailment_score >= NLI_ENTAILMENT_THRESHOLD and entailment_score > float(scores[0]),
        "score": entailment_score,
        "contradiction_score": float(scores[0]),
    }


def check_all_claims(claims: list[str], premise: str) -> list[dict]:
    enc = _get_encoder()
    if enc is None:
        return [_safe_result(c) for c in claims]
    if len(premise) > _MAX_PREMISE_CHARS:
        logger.warning("NLI premise truncated from %d to %d chars", len(premise), _MAX_PREMISE_CHARS)
        premise = premise[:_MAX_PREMISE_CHARS]
    pairs = [[premise, c] for c in claims]
    all_scores = enc.predict(pairs)
    results = []
    for claim, scores in zip(claims, all_scores):
        if len(scores) < 3:
            logger.warning("NLI model returned unexpected score shape: %s", scores)
            results.append(_safe_result(claim))
            continue
        entailment_score = float(scores[2])
        results.append({
            "claim": claim,
            "entailed": entailment_score >= NLI_ENTAILMENT_THRESHOLD and entailment_score > float(scores[0]),
            "score": entailment_score,
            "contradiction_score": float(scores[0]),
        })
    return results
