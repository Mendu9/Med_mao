import logging
from mao.core.config import NLI_MODEL, NLI_ENTAILMENT_THRESHOLD

logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(NLI_MODEL)
    return _cross_encoder


def check_claim(premise: str, claim: str) -> dict:
    enc = _get_encoder()
    scores = enc.predict([[premise, claim]])[0]
    entailment_score = float(scores[2])
    return {
        "claim": claim,
        "entailed": entailment_score >= NLI_ENTAILMENT_THRESHOLD,
        "score": entailment_score,
        "contradiction_score": float(scores[0]),
    }


def check_all_claims(claims: list[str], premise: str) -> list[dict]:
    enc = _get_encoder()
    pairs = [[premise, c] for c in claims]
    all_scores = enc.predict(pairs)
    results = []
    for claim, scores in zip(claims, all_scores):
        entailment_score = float(scores[2])
        results.append({
            "claim": claim,
            "entailed": entailment_score >= NLI_ENTAILMENT_THRESHOLD,
            "score": entailment_score,
            "contradiction_score": float(scores[0]),
        })
    return results
