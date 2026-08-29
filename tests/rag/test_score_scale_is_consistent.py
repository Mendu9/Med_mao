"""A relevance score must mean the same thing on every request — or say it doesn't.

Found during Wave 7 remediation, by noticing that a live `graph.invoke` returned
`retrieved_docs` with scores of 27.17 and 25.52 while every threshold in the
system compares against 0.10 or 0.20.

`rerank()` has a passthrough branch for when the cross-encoder is unavailable —
disabled by `MAO_DISABLE_RERANKER`, not installed, or failed to load. It returned
the upstream retrieval score untouched, under a comment reading "Sort by existing
cosine score". That comment is why this survived: the score is a cosine only when
the *vector* leg produced it. The BM25 leg produces raw Okapi scores, which are
unbounded and routinely 20-30.

Three things downstream then silently stopped meaning anything:

  - `graphrag_agent.RAG_CONFIDENCE_THRESHOLD` (0.10) and `clinical_agent`'s 0.20
    gate the PubMed and web fallback. Against a BM25 score they are always true,
    so the fallback that exists for "the corpus does not cover this" could not
    fire on any query BM25 ranked first.
  - `metadata["top_rag_score"]` and `rag_sufficient` are returned to the client.
  - `RetrievalTrace.top_score` is on every trace. Phase 1 is "Architecture
    Stabilization and Learning Data Plane", and a stored metric whose scale
    depends on whether a model happened to load is not learnable.

The fix is NOT to rescale. Min-max over the result set was the obvious move and
it is wrong: it maps the best chunk to 1.0 whatever its absolute quality, so the
threshold would still never fire — the defect would survive the fix while looking
repaired.

An unbounded BM25 score and a cross-encoder probability are genuinely not
comparable, so the honest answer is to record WHICH scorer produced the number
and let the gate apply only to the scorer it was calibrated against. When the
score is uncalibrated the confidence is unknown, and the safe reading of
"unknown" is to fetch more evidence, not less — so the fallback fires.
"""
from __future__ import annotations

import pytest

from mao.rag.reranker import CROSS_ENCODER, UNCALIBRATED, rerank


@pytest.fixture
def reranker_off(monkeypatch: pytest.MonkeyPatch):
    """Force the passthrough branch — the degraded mode this is all about."""
    monkeypatch.setenv("MAO_DISABLE_RERANKER", "1")


def _chunks(*scores: float) -> list[dict]:
    return [
        {"text": f"chunk {i}", "score": s, "chunk_id": f"c{i}"}
        for i, s in enumerate(scores)
    ]


class TestAScoreDeclaresWhoProducedIt:
    def test_passthrough_scores_are_marked_uncalibrated(self, reranker_off) -> None:
        ranked = rerank("q", _chunks(27.17, 25.52), top_k=2)
        assert all(r.scorer == UNCALIBRATED for r in ranked)

    def test_passthrough_preserves_the_raw_value(self, reranker_off) -> None:
        """Rescaling would invent a precision the number does not have."""
        ranked = rerank("q", _chunks(27.17, 25.52), top_k=2)
        assert ranked[0].score == pytest.approx(27.17)

    def test_ordering_is_still_best_first(self, reranker_off) -> None:
        ranked = rerank("q", _chunks(24.28, 27.17, 25.52), top_k=3)
        assert [r.metadata["chunk_id"] for r in ranked] == ["c1", "c2", "c0"]

    def test_no_chunks_is_still_empty(self, reranker_off) -> None:
        assert rerank("q", [], top_k=3) == []

    def test_metadata_survives(self, reranker_off) -> None:
        ranked = rerank("q", _chunks(27.17, 1.0), top_k=2)
        assert ranked[0].metadata["chunk_id"] == "c0"
        assert ranked[0].text == "chunk 0"

    def test_the_default_is_uncalibrated(self) -> None:
        """A chunk from anywhere else must not claim calibration it lacks —
        including one rebuilt from a cache entry written before this field
        existed."""
        from mao.rag.reranker import RankedChunk

        assert RankedChunk(text="t", score=0.5, metadata={}).scorer == UNCALIBRATED


class TestAnUncalibratedScoreDoesNotSatisfyTheConfidenceGate:
    """Fail-safe direction: unknown confidence fetches MORE evidence."""

    def test_graphrag_treats_an_uncalibrated_score_as_insufficient(self) -> None:
        from mao.rag.reranker import RankedChunk
        from mao.agents.graphrag_agent import rag_is_sufficient

        chunks = [RankedChunk(text="t", score=27.17, metadata={}, scorer=UNCALIBRATED)]
        assert rag_is_sufficient(chunks) is False, (
            "a raw BM25 score of 27.17 cleared a threshold of 0.10, so the "
            "PubMed and web fallback could never fire"
        )

    def test_a_calibrated_score_above_the_threshold_is_sufficient(self) -> None:
        from mao.rag.reranker import RankedChunk
        from mao.agents.graphrag_agent import rag_is_sufficient

        chunks = [RankedChunk(text="t", score=0.85, metadata={}, scorer=CROSS_ENCODER)]
        assert rag_is_sufficient(chunks) is True

    def test_a_calibrated_score_below_the_threshold_is_not_sufficient(self) -> None:
        from mao.rag.reranker import RankedChunk
        from mao.agents.graphrag_agent import rag_is_sufficient

        chunks = [RankedChunk(text="t", score=0.02, metadata={}, scorer=CROSS_ENCODER)]
        assert rag_is_sufficient(chunks) is False

    def test_no_chunks_is_not_sufficient(self) -> None:
        from mao.agents.graphrag_agent import rag_is_sufficient

        assert rag_is_sufficient([]) is False


class TestTheTraceCanInterpretItsOwnScore:
    def test_the_scorer_reaches_the_retrieval_trace(self) -> None:
        """A stored `top_score` is only learnable if its scale is recoverable."""
        from mao.api.tracing import build_trace

        trace = build_trace(
            trace_id="t",
            result={
                "intent": "graphrag",
                "metadata": {
                    "chunks_retrieved": 3,
                    "top_scores": [0.81],
                    "score_scorer": CROSS_ENCODER,
                    "sources": [{}],
                },
            },
            latency_ms=1.0,
        )
        assert trace.retrieval is not None
        assert trace.retrieval.scorer == CROSS_ENCODER
