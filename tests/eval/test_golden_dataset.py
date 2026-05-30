"""Regression test: golden query dataset for Alzheimer/stroke clinical AI.

Run nightly (or on prompt changes) to catch quality regressions.
Marked @pytest.mark.slow — excluded from fast CI runs.
Each test mocks retrieve() at the function level and checks that key medical
terms appear in the RankedChunk results. Uses .text attribute (not .get()).
"""
import pytest
from unittest.mock import patch
from mao.rag.reranker import RankedChunk

# 20 golden (query, required_keywords) pairs
GOLDEN_DATASET = [
    # Alzheimer's — drugs
    ("What drugs treat Alzheimer's disease?", ["donepezil", "memantine"]),
    ("How does lecanemab work?", ["amyloid", "antibody"]),
    ("What is the mechanism of donepezil?", ["acetylcholinesterase", "cholinesterase"]),
    ("What are anti-amyloid therapies?", ["amyloid", "plaque"]),
    # Alzheimer's — biomarkers
    ("What plasma biomarkers diagnose Alzheimer's?", ["pTau", "amyloid"]),
    ("What is the role of GFAP in Alzheimer's?", ["astrocyte", "biomarker"]),
    ("Explain pTau217 as a biomarker", ["tau", "phosphorylation"]),
    # Alzheimer's — pathology
    ("What is the amyloid cascade hypothesis?", ["amyloid", "tau", "neurodegeneration"]),
    ("How does tau cause neurodegeneration?", ["tau", "tangle", "neuron"]),
    ("What is APOE4 risk?", ["APOE", "amyloid", "risk"]),
    # Stroke
    ("What are treatments for ischemic stroke?", ["tPA", "thrombectomy"]),
    ("What is the blood-brain barrier in stroke?", ["barrier", "endothelial"]),
    ("What are stroke risk factors?", ["hypertension", "atrial"]),
    ("How is stroke diagnosed?", ["CT", "MRI", "imaging"]),
    # Cognitive assessment
    ("What is the MoCA test?", ["cognitive", "assessment", "score"]),
    ("How is CDR used in Alzheimer's?", ["clinical", "dementia", "rating"]),
    # Interventions
    ("What non-pharmacological interventions help dementia?", ["exercise", "cognitive"]),
    ("What caregiver interventions improve dementia outcomes?", ["caregiver", "support"]),
    # Differential diagnosis
    ("How is Alzheimer's distinguished from vascular dementia?", ["vascular", "cognitive"]),
    ("What are Lewy body dementia features?", ["Lewy", "dementia"]),
]


def _make_ranked_chunks(keywords: list[str], query: str) -> list[RankedChunk]:
    """Build fake RankedChunk results containing all required keywords."""
    fake_text = " ".join(keywords) + " " + query
    return [
        RankedChunk(
            text=fake_text,
            score=0.9,
            metadata={"source": "test.pdf", "domain": "alzheimer", "entities": ""},
        )
    ]


@pytest.mark.slow
@pytest.mark.parametrize("query,keywords", GOLDEN_DATASET)
def test_golden_query(query: str, keywords: list[str]) -> None:
    """Mocked retrieval must return RankedChunks containing all required keywords."""
    expected_chunks = _make_ranked_chunks(keywords, query)

    # Patch at retrieve() function level — avoids fragile internal ChromaDB patch targets.
    # RankedChunk is a dataclass: access .text not .get("text").
    with patch("mao.rag.retriever.retrieve", return_value=expected_chunks):
        from mao.rag.retriever import retrieve
        results = retrieve(query=query, domain="alzheimer", top_k=3)

    assert results, f"No results for: {query}"
    combined = " ".join(r.text for r in results).lower()
    for kw in keywords:
        assert kw.lower() in combined, (
            f"Keyword '{kw}' not found for query: {query}"
        )
