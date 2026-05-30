"""Semantic triple extractor for biomedical text using Groq LLM.

Extracts typed subject-predicate-object triples to replace co-occurrence edges
with meaningful semantic relationships in the entity graph.
"""
from __future__ import annotations
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

VALID_RELATIONS = frozenset([
    "treats", "causes", "associated_with", "inhibits",
    "biomarker_of", "part_of", "interacts_with", "expressed_in",
    "increases_risk_of", "prevents",
])

_TRIPLE_PROMPT = """\
Extract biomedical relationships from the text below as triples.
Use ONLY these relation types: treats, causes, associated_with, inhibits, biomarker_of, part_of, interacts_with, expressed_in, increases_risk_of, prevents.

Output format (one triple per line, nothing else):
ENTITY_A | relation | ENTITY_B

Rules:
- Only include high-confidence, factual relationships
- Both entities must be biomedical concepts (disease, drug, gene, protein, symptom)
- Use the exact relation words listed above
- Maximum 10 triples per text

Text:
{text}

Triples:"""


def extract_triples(text: str, model: str | None = None) -> list[tuple[str, str, str]]:
    """Extract (subject, predicate, object) triples from biomedical text via Groq.

    Returns list of (entity_A, relation, entity_B) tuples.
    Returns empty list if extraction fails.
    """
    if not text or len(text.strip()) < 50:
        return []

    try:
        from mao.core.llm import chat
        from mao.core.config import FAST_MODEL
    except ImportError:
        logger.warning("mao.core.llm not available; skipping triple extraction")
        return []

    _model = model or FAST_MODEL
    prompt = _TRIPLE_PROMPT.format(text=text[:1500])  # cap to avoid token overflow

    try:
        response = chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            model=_model,
        )
        return _parse_triples(response)
    except Exception as exc:
        logger.debug("Triple extraction failed: %s", exc)
        return []


def _parse_triples(response: str) -> list[tuple[str, str, str]]:
    """Parse LLM response into (subject, predicate, object) tuples."""
    triples: list[tuple[str, str, str]] = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        subj, pred, obj = parts
        pred_lower = pred.lower().replace(" ", "_").replace("-", "_")
        # Normalize relation to valid set
        matched = next((r for r in VALID_RELATIONS if r in pred_lower), None)
        if matched and subj and obj:
            triples.append((subj, matched, obj))
    return triples


def add_triples_to_graph(
    G: Any,  # nx.MultiDiGraph
    triples: list[tuple[str, str, str]],
    doc_id: str,
) -> int:
    """Add extracted triples as directed typed edges to the graph.

    Adds nodes if they don't exist. Returns count of edges added.
    """
    count = 0
    for subj, pred, obj in triples:
        if not G.has_node(subj):
            G.add_node(subj, name=subj, node_type="entity", source=doc_id, edge_type="extracted")
        if not G.has_node(obj):
            G.add_node(obj, name=obj, node_type="entity", source=doc_id, edge_type="extracted")
        G.add_edge(
            subj, obj,
            relation=pred,
            source=doc_id,
            edge_type="semantic",
        )
        count += 1
    return count
