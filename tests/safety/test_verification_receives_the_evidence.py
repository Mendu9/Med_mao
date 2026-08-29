"""The verification chain must actually be given the evidence it verifies against.

Found during Wave 7 remediation by tracing what `council_node` and
`verification_node` really receive from a live `graph.invoke`. Neither exit-gate
review reported either of these, because both examined whether the controls
*ran*, not what they were handed.

DEFECT A — the premise is a Python dict repr, truncated mid-literal.
`graphrag_node` writes `retrieved_docs` as `list[dict]`. `council_node` builds
its context with `str(c)`, and `verification._premise_from` falls back to
`str(doc)[:300]`. The judge's premise was literally:

    {'text': 'Figure 6 shows that one subject of LMCI was misclassified as EMCI
    as in the case of five multiclass classifications while five LMCI were
    incorrectly diag {'text': 'From a computational point of view, link ...

Four truncated reprs concatenated. About 40 characters of every 300-character
budget went on `{'text': '`, the `source` key was cut off every time so no
attribution ever reached the judge, and each excerpt ended mid-word. A judge
scoring groundedness against that is scoring noise — observed live returning
`groundedness: 0` and notes describing a topic that was not in the corpus.

DEFECT B — on the CLINICAL route the premise is empty entirely.
`clinical_node` never writes `retrieved_docs`; it keeps its chunks under
`_ranked_chunks` in `result_meta`, which is stripped by the `_` prefix rule. So
for the highest-risk route in the system:

  - `_run_nli` gets no premise and returns `[]`, so the NLI gate cannot fire;
  - `_members_for("")` narrows the council to the safety member alone, so the
    accuracy and hallucination vetoes never run;
  - the judge scores groundedness against nothing.

The HIGH-risk path had the weakest verification of any route.

Both are fixed by carrying typed `Evidence` between retrieval and verification
rather than whatever shape each agent happened to leave behind — which is also
what makes `mao/schemas/evidence.py` a consumed contract rather than a declared
one.
"""
from __future__ import annotations

import pytest

from mao.safety.verification import _premise_from
from mao.schemas.evidence import Evidence


def _evidence(text: str, source: str = "pmc:PMC1", score: float = 0.9) -> Evidence:
    return Evidence(
        evidence_id=f"{source}#0", text=text, source_id=source, score=score
    )


# ---------------------------------------------------------------------------
# DEFECT A — the premise is readable prose
# ---------------------------------------------------------------------------

class TestThePremiseIsProseNotARepr:
    def test_a_dict_shaped_doc_yields_its_text_not_its_repr(self) -> None:
        premise = _premise_from([{"text": "Donepezil inhibits acetylcholinesterase.",
                                  "source": "pmc:PMC1", "score": 0.9}])
        assert premise.strip() == "Donepezil inhibits acetylcholinesterase."

    def test_no_python_literal_syntax_leaks_into_the_premise(self) -> None:
        docs = [{"text": f"Finding {i}.", "source": "s", "score": 0.5} for i in range(4)]
        premise = _premise_from(docs)
        for artefact in ("{'text'", "'score'", "'source'", "}"):
            assert artefact not in premise, f"{artefact!r} leaked into the premise"

    def test_typed_evidence_yields_its_text(self) -> None:
        premise = _premise_from([_evidence("Amyloid plaques accumulate in cortex.")])
        assert premise.strip() == "Amyloid plaques accumulate in cortex."

    def test_objects_with_a_text_attribute_still_work(self) -> None:
        class _Chunk:
            text = "Retrieved chunk text."

        assert _premise_from([_Chunk()]).strip() == "Retrieved chunk text."

    def test_plain_strings_still_work(self) -> None:
        assert "a chunk" in _premise_from(["a chunk"])

    def test_the_budget_is_spent_on_text_not_punctuation(self) -> None:
        """A 300-char doc budget must yield ~300 chars of document."""
        long_text = "A" * 500
        premise = _premise_from([{"text": long_text, "source": "s", "score": 0.1}])
        assert premise.count("A") >= 290


# ---------------------------------------------------------------------------
# DEFECT B — the clinical route publishes its evidence
# ---------------------------------------------------------------------------

class TestTheClinicalRoutePublishesItsEvidence:
    @pytest.fixture
    def clinical_state(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        from mao.agents import clinical_agent
        from mao.core.state import make_initial_state

        class _Chunk:
            def __init__(self, text: str, source: str, score: float) -> None:
                self.text = text
                self.score = score
                self.metadata = {"source": source, "chunk_id": "c1", "title": source}

        monkeypatch.setattr(
            clinical_agent,
            "retrieve",
            lambda *a, **k: [
                _Chunk("Donepezil is an acetylcholinesterase inhibitor.", "pmc:PMC1", 0.9),
                _Chunk("Memantine is an NMDA receptor antagonist.", "pmc:PMC2", 0.8),
            ],
        )
        monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
        monkeypatch.setattr(clinical_agent, "_call_llm", lambda s, u: "An answer.")

        state = make_initial_state("what treats alzheimer's?", "u1")
        return dict(clinical_agent.clinical_node(state))

    def test_retrieved_docs_is_populated(self, clinical_state) -> None:
        assert clinical_state.get("retrieved_docs"), (
            "the clinical route left retrieved_docs empty, so the NLI gate, the "
            "accuracy and hallucination council members and the judge's premise "
            "all had nothing to work from on the highest-risk route"
        )

    def test_the_evidence_carries_its_text(self, clinical_state) -> None:
        premise = _premise_from(clinical_state["retrieved_docs"])
        assert "acetylcholinesterase" in premise

    def test_the_council_would_seat_every_member(self, clinical_state) -> None:
        from mao.agents.llm_council import _members_for

        context = "\n".join(str(c) for c in clinical_state["retrieved_docs"][:4])
        assert set(_members_for(context)) == {"accuracy", "hallucination", "safety"}, (
            "an empty context narrows the council to the safety member alone"
        )

    def test_no_ranked_chunk_objects_leak_into_the_response_metadata(
        self, clinical_state
    ) -> None:
        assert not any(k.startswith("_") for k in clinical_state["metadata"])


class TestTheGraphRouteAlsoPublishesEvidence:
    def test_graphrag_evidence_reaches_verification(self, graph_harness) -> None:
        """Through the real graph, not through a hand-built state."""
        graph_harness.retrieved = [
            type("C", (), {"text": "Donepezil raises acetylcholine.", "score": 0.9,
                           "metadata": {"source": "pmc:PMC1", "chunk_id": "c1"}})()
        ]
        result = graph_harness.invoke("what does donepezil do?")
        premise = _premise_from(result.get("retrieved_docs") or [])
        assert "acetylcholine" in premise
        assert "{'text'" not in premise
