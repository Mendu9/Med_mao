"""Wave 11 / ADV12-7 — the retrieval fallbacks are egress sinks too.

`tests/api/test_phi_never_reaches_the_provider.py` binds a recorder at the model
gateway and `tests/api/test_no_raw_query_leaves_the_process.py` binds one at the
Groq SDK, the application log and the database. Between them they cover four
sinks — and `graphrag_node` has three more that neither of them can see:

    graphrag_agent:157  `_live_pubmed_search(user_query)`   -> NCBI, over HTTP
    graphrag_agent:165  `_web_search(user_query)`           -> public engines
    graphrag_agent:134  `retrieve(user_query, ...)`         -> the vector store

An adversarial review found these by running the process and watching the
socket: the question was transmitted in plaintext URLs to five public search
engines. Both existing PHI tests stub `_web_search_clinical` and neither binds a
recorder at any of the three, so a leak here was invisible to the whole suite.

What reaches them is `state["user_query"]`, which IS the scrubbed string
(`main.py:342` sets `user_query=safe_query`), so the design is sound — searching
the web with a de-identified clinical question is the intended behaviour. That
is exactly why it needs a test: the guarantee is inherited from the scrubber,
and an inherited guarantee that nobody asserts is the shape of every defect this
project has shipped.
"""
from __future__ import annotations

import pytest

from mao.agents import graphrag_agent

PATIENT_NAME = "Arthur Neville Kowalczyk"
MRN = "LDS/9931/C"
NHS_NUMBER = "943 476 5919"
PHI = (PATIENT_NAME, MRN, NHS_NUMBER)

QUERY_WITH_PHI = (
    f"Patient Name: {PATIENT_NAME}\n"
    f"MRN: {MRN}\n"
    f"NHS Number: {NHS_NUMBER}\n"
    "Should donepezil be titrated in a patient with sinus bradycardia?"
)


class EgressRecorder:
    """Every string handed to a retrieval sink that leaves this process."""

    def __init__(self) -> None:
        self.sent: dict[str, list[str]] = {}

    def record(self, sink: str, value: str) -> None:
        self.sent.setdefault(sink, []).append(value)

    def leaked(self) -> dict[str, list[str]]:
        return {
            sink: [identifier for identifier in PHI if identifier in "\n".join(values)]
            for sink, values in self.sent.items()
            if any(identifier in "\n".join(values) for identifier in PHI)
        }


def _stub_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the node offline. The gateway is a sink the OTHER tests cover."""
    from mao.providers.llm.base import ProviderResponse

    monkeypatch.setattr(
        graphrag_agent.gateway,
        "complete",
        lambda **kwargs: ProviderResponse(text="An answer.", input_tokens=1, output_tokens=1),
    )


@pytest.fixture
def egress(monkeypatch: pytest.MonkeyPatch) -> EgressRecorder:
    """Bind a recorder to all three retrieval sinks. Nothing leaves the host."""
    recorder = EgressRecorder()

    def fake_web_search(query, num_results=5):
        recorder.record("web_search", str(query))
        return []

    def fake_pubmed(query, max_results=5):
        recorder.record("pubmed", str(query))
        return []

    def fake_retrieve(query, **kwargs):
        recorder.record("vector_store", str(query))
        return []

    monkeypatch.setattr(graphrag_agent, "_web_search", fake_web_search)
    monkeypatch.setattr(graphrag_agent, "_live_pubmed_search", fake_pubmed)
    monkeypatch.setattr(graphrag_agent, "retrieve", fake_retrieve)
    return recorder


class TestTheRetrievalSinksNeverSeeAnIdentifier:
    def test_the_scrubbed_query_is_what_reaches_them(
        self, egress: EgressRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`graphrag_node` reads `state["user_query"]`, and the route puts the
        SCRUBBED query there. Asserted at the sinks, not at the assignment."""
        from mao.core.pii_scrubber import scrub_pii

        _stub_synthesis(monkeypatch)
        graphrag_agent.graphrag_node(
            {"user_query": scrub_pii(QUERY_WITH_PHI), "intent": "graphrag"}
        )

        assert egress.sent, "no retrieval sink was reached — the probe is vacuous"
        assert egress.leaked() == {}, (
            f"identifiers reached a retrieval sink: {egress.leaked()}"
        )

    def test_the_clinical_question_still_reaches_them(
        self, egress: EgressRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise the test above passes because retrieval got nothing."""
        from mao.core.pii_scrubber import scrub_pii

        _stub_synthesis(monkeypatch)
        graphrag_agent.graphrag_node(
            {"user_query": scrub_pii(QUERY_WITH_PHI), "intent": "graphrag"}
        )

        everything = "\n".join(
            value for values in egress.sent.values() for value in values
        )
        assert "bradycardia" in everything
        assert "donepezil" in everything

    def test_a_raw_query_would_be_caught(self, egress: EgressRecorder) -> None:
        """The recorder must be able to FAIL, or it proves nothing.

        This is the mutation the other two tests cannot perform on themselves:
        it feeds the node an UNSCRUBBED query and asserts the probe notices.
        """
        graphrag_agent.graphrag_node(
            {"user_query": QUERY_WITH_PHI, "intent": "graphrag"}
        )
        assert egress.leaked(), (
            "the recorder did not notice raw PHI at any sink — it is blind"
        )
