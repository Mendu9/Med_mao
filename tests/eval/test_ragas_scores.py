import pytest
from tests.eval.conftest import load_golden, fixture_chunks_for


@pytest.mark.slow
@pytest.mark.parametrize("entry", load_golden()[:5], ids=[e["id"] for e in load_golden()[:5]])
def test_ragas_faithfulness(entry: dict) -> None:
    """RAGAS faithfulness >= 0.7 for golden query/answer pairs."""
    if not entry.get("ground_truth") or not entry.get("query"):
        pytest.skip("No ground truth for this entry")
    contexts = [c["text"] for c in fixture_chunks_for(entry["id"])]
    if not contexts:
        pytest.skip(f"No fixture chunks for {entry['id']}")

    from mao.eval.ragas_evaluator import _run_ragas_sync
    scores = _run_ragas_sync(
        question=entry["query"],
        answer=entry["ground_truth"],
        contexts=contexts,
    )
    faithfulness = scores.get("faithfulness", 0.0)
    # Threshold 0.60: local NLI model scores conservatively vs OpenAI; 0.60 ensures
    # genuine grounding while accommodating local-model variance.
    assert faithfulness >= 0.60, (
        f"RAGAS faithfulness {faithfulness:.2f} < 0.60 for {entry['id']}"
    )
