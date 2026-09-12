r"""ADV16-3 and A-4, at the boundaries that failed.

Two reviews found two non-overlapping halves of one root cause, and the union
is what this file asserts.

    A-4      the gateway was wired to 1 of its 7 declared destinations. Three
             `authorise()` call sites existed, all MODEL_PROVIDER, all in
             `mao/providers/gateway.py`. Web search, the scholarly API, the
             vector store and external memory reached the network with no
             `authorise()` call at all, so the `(EXTERNAL_MEMORY, MEMORY_WRITE)`
             row and the three `EVIDENCE_SEARCH` rows were rows nothing read.

    ADV16-3  `mao/api/main.py` fired `_score_response_async` on the /chat
             REQUEST path, reaching `ragas_evaluator`, which imports
             `langchain_groq.ChatGroq` - and `ChatGroq.validate_environment`
             constructs its OWN `groq.Groq` and `groq.AsyncGroq`. Measured on
             one ordinary /chat: 45 SDK `create()` calls against 8 `authorise()`
             calls, reproduced at 36/44, 39/47 and 37/45, with the sync count
             equal to the authorise count exactly and every async call
             unauthorised.

The table lookup is not the part that matters most. `authorise()` is also where
the run-scoped `RequestProtection.leaked_in` assertion runs - the control
`gateway` calls "the one that closes the class of defect the previous six waves
kept reopening". On every bypassing path that second wall did not exist, so a
de-identification defect that put a patient's name into a retrieval query was
refused on its way to the synthesis model and sent to the web-search provider
and to NCBI without comment.

Why the phase's own tests could not see ADV16-3: the route-level PHI tests bind
a recorder at the provider seam the GATEWAY uses. A client built inside a
third-party library is a different object, so an assertion bound to
`mao.providers` cannot observe it. The counter below is therefore installed at
CLASS level on `groq.resources.chat.completions`, which catches every instance
including one a library builds for itself.
"""
from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from mao.trust.classes import InputChannel
from mao.trust.egress.gateway import (
    EgressRefused,
    RequestProtection,
    protected_request,
)


@pytest.fixture()
def unowned_sdk_calls(monkeypatch):
    """Every provider call the EgressGateway does not own, counted.

    The design matters, because the obvious version of this test cannot run on
    this host. Counting raw SDK calls and comparing them against `authorise()`
    calls - the reviewer's measurement, 45 against 8 - needs a working provider:
    stub the SDK to raise and the graph degrades, produces no sources, never
    reaches the branch that carried the unauthorised client, and the counter
    reports a comfortable zero on an unfixed build.

    So the two seams are separated instead of counted against each other. A
    `RecordingProvider` is bound at `gateway.set_provider`, which is where every
    AUTHORISED call goes, and it never touches the groq SDK. The groq SDK is
    then instrumented at CLASS level on both `Completions.create` and
    `AsyncCompletions.create`, so it captures every client instance including
    one a third-party library builds for itself.

    Anything arriving at the SDK is therefore, by construction, a client the
    gateway does not own. The assertion is that there are none - which is the
    same property the reviewer measured, stated in a form that does not depend
    on a populated vector store or a reachable provider.
    """
    calls: list[str] = []

    import groq.resources.chat.completions as completions_module

    def _sync(self, **kwargs):
        calls.append("sync")
        raise RuntimeError("an unowned provider client reached the SDK")

    async def _async(self, **kwargs):
        calls.append("async")
        raise RuntimeError("an unowned provider client reached the SDK")

    monkeypatch.setattr(completions_module.Completions, "create", _sync)
    monkeypatch.setattr(completions_module.AsyncCompletions, "create", _async)
    return calls


@pytest.fixture()
def gateway_provider():
    """The real gateway seam, recording. Authorised calls end here."""
    from mao.providers import gateway

    from .recorders import RecordingProvider

    recorder = RecordingProvider()
    gateway.set_provider(recorder)
    try:
        yield recorder
    finally:
        gateway.reset_provider()


@pytest.fixture()
def retrieval_returns_sources(monkeypatch):
    """Make the request produce `sources`, which is what ADV16-3 needs.

    `main.py` fired the scorer only `if contexts:`, and `contexts` is built
    from `metadata["sources"]`. On a host with no vector store retrieval
    returns nothing, so the branch never runs and a counter bound to it passes
    while measuring nothing at all. The reviewer's run had a populated store;
    this fixture is how that condition is reproduced here rather than assumed.
    """
    from mao.agents import clinical_agent, graphrag_agent

    class _Chunk:
        text = "NICE NG97 advises an ECG before starting a cholinesterase inhibitor."
        score = 0.9
        metadata = {"source": "NG97", "chunk_id": "c1"}

    for module in (clinical_agent, graphrag_agent):
        if hasattr(module, "retrieve"):
            monkeypatch.setattr(module, "retrieve", lambda *a, **k: [_Chunk()])
    return _Chunk


def _chat(client: TestClient):
    return client.post(
        "/chat", json={"query": "Is donepezil safe in bradycardia?", "user_id": "t"}
    )


class TestNoProviderCallEscapesTheGateway:
    def test_no_unowned_client_reaches_the_provider_on_a_chat_request(
        self, unowned_sdk_calls, gateway_provider, retrieval_returns_sources
    ) -> None:
        from mao.api.main import app

        with TestClient(app) as client:
            _chat(client)

        assert unowned_sdk_calls == [], (
            f"{len(unowned_sdk_calls)} provider call(s) reached the groq SDK "
            f"from a client the gateway does not own ({set(unowned_sdk_calls)}). "
            "Such a client has no destination, no purpose, no trust class, and "
            "no run-scoped identifier assertion."
        )

    def test_the_same_holds_on_the_streaming_route(
        self, unowned_sdk_calls, gateway_provider, retrieval_returns_sources
    ) -> None:
        """`/chat` and `/chat/stream` have drifted apart twice already."""
        from mao.api.main import app

        with TestClient(app) as client:
            client.post(
                "/chat/stream",
                json={"query": "Is donepezil safe in bradycardia?", "user_id": "t"},
            )

        assert unowned_sdk_calls == []

    def test_the_request_really_exercised_the_gateway(
        self, unowned_sdk_calls, gateway_provider, retrieval_returns_sources
    ) -> None:
        """Non-vacuity. If the request made no provider call at all, the
        assertions above would pass on any build, fixed or not."""
        from mao.api.main import app

        with TestClient(app) as client:
            _chat(client)

        assert gateway_provider.calls, (
            "no call reached the gateway seam either - this probe is not "
            "exercising the request path"
        )

    def test_the_counter_can_see_a_client_built_inside_a_library(
        self, unowned_sdk_calls
    ) -> None:
        """The other half of non-vacuity, and the ADV16-3 mechanism itself.

        `langchain_groq.ChatGroq.validate_environment` constructs its OWN
        `groq.Groq` and `groq.AsyncGroq`. Those are not
        `mao.core.llm._get_groq_client`, do not go through
        `mao.providers.gateway`, and are invisible to any recorder bound at
        `mao.providers` - which is precisely why the phase's own route-level
        PHI tests were green while 37 of 45 calls bypassed the gateway.

        This proves the class-level hook DOES see such a client, so the empty
        lists above are a fact about the request path rather than about the
        instrument.
        """
        pytest.importorskip("langchain_groq")
        from langchain_groq import ChatGroq

        with pytest.raises(Exception):  # noqa: B017 - the stub raises by design
            ChatGroq(api_key="x", model="openai/gpt-oss-20b", temperature=0).invoke(
                "hello"
            )

        assert unowned_sdk_calls, (
            "a client built inside langchain_groq did NOT reach the counter, "
            "so this file cannot detect the ADV16-3 bypass at all"
        )


class TestRagasIsNotOnTheRequestPath:
    """ADV16-4: the module asserts this about itself in source, and it was
    false. A claim about the CALLER must be verified from the caller."""

    def test_no_request_route_reaches_the_ragas_evaluator(self) -> None:
        source = pathlib.Path("mao/api/main.py").read_text(encoding="utf-8")
        offending = [
            line
            for line in source.splitlines()
            if "ragas" in line.lower() and not line.strip().startswith("#")
        ]
        assert offending == [], (
            "mao/api/main.py still reaches the ragas evaluator, which builds "
            f"its own provider client outside the gateway: {offending}"
        )

    def test_the_double_scrub_is_closed_by_the_caller_not_by_the_scrub(self) -> None:
        """The residual the adversarial review folded into ADV16-3, closed in
        the direction that does not weaken anything.

        At b63311d this module was the last reachable site where text that had
        already been through the protected boundary was scrubbed a second time,
        because `main.py` passed `safe_query` into it on the request path. A
        second pass can over-redact a clinical line, which is the whole reason
        `boundary.py` says "there is no second pass".

        Deleting `scrub_pii` from this module would also close it, and would be
        wrong: the scrub is a CONTRACT on a parameter that previously
        documented itself as "the user's original question", and an offline
        caller may pass raw text for which it is the only control. What makes
        the second pass unreachable is that the already-protected caller is
        gone, which is asserted above.
        """
        source = pathlib.Path("mao/eval/ragas_evaluator.py").read_text(
            encoding="utf-8"
        )
        live = [
            line
            for line in source.splitlines()
            if "scrub_pii(" in line and not line.strip().startswith("#")
        ]
        assert live, (
            "the module-level de-identification contract was removed. The "
            "double-scrub is closed by removing the request-path caller, not "
            "by removing this."
        )


class TestTheSearchSinksAreBehindTheWall:
    """A-4. Authorised at the SINK, not at the agent: `_web_search_clinical` is
    one caller of `web_search`, and an authorise() there would leave the next
    caller unguarded - the "missing call site on a second channel" shape this
    phase exists to end."""

    def test_web_search_refuses_a_query_carrying_a_removed_identifier(self) -> None:
        from mao.core.web_search import web_search

        protection = RequestProtection(trace_id="t")
        protection.record_identifier("NAME", "Harold Nkemdirim", InputChannel.QUERY)
        with protected_request(protection):
            with pytest.raises(EgressRefused):
                web_search("bradycardia in Harold Nkemdirim", num_results=1)

    def test_pubmed_refuses_a_query_carrying_a_removed_identifier(self) -> None:
        from mao.data.ingest_pubmed import live_pubmed_search

        protection = RequestProtection(trace_id="t")
        protection.record_identifier("MRN", "RGT/44219/B", InputChannel.REPORT)
        with protected_request(protection):
            with pytest.raises(EgressRefused):
                live_pubmed_search("donepezil RGT/44219/B", max_results=1)

    def test_a_clean_web_query_is_authorised_and_recorded(self, monkeypatch) -> None:
        """Non-vacuity: the guard must pass traffic it should pass, and the
        decision must be visible in the trace."""
        from mao.core import web_search as module

        monkeypatch.setattr(module, "_ddg_search", lambda q, n: [])
        monkeypatch.setattr(module, "_brave_search", lambda q, n: [])
        monkeypatch.setattr(module, "_serpapi_search", lambda q, n: [])

        protection = RequestProtection(trace_id="t")
        with protected_request(protection):
            module.web_search("donepezil bradycardia", num_results=1)

        assert ("web_search", "evidence_search", "safe_derived_text") in (
            protection.authorised
        )


class TestTheVectorStoreIsBehindTheWall:
    def test_a_query_carrying_a_removed_identifier_is_refused(self) -> None:
        from mao.rag import retriever

        protection = RequestProtection(trace_id="t")
        protection.record_identifier("NAME", "Harold Nkemdirim", InputChannel.QUERY)
        with protected_request(protection):
            with pytest.raises(EgressRefused):
                retriever._vector_search("Harold Nkemdirim bradycardia", 1)

    def test_an_entity_search_is_authorised_too(self) -> None:
        from mao.rag import retriever

        protection = RequestProtection(trace_id="t")
        protection.record_identifier("NAME", "Harold Nkemdirim", InputChannel.QUERY)
        with protected_request(protection):
            with pytest.raises(EgressRefused):
                retriever._entity_search(["Harold Nkemdirim"], 1)

    def test_a_clean_query_records_the_authorisation(self) -> None:
        from mao.rag import retriever

        protection = RequestProtection(trace_id="t")
        with protected_request(protection):
            try:
                retriever._vector_search("donepezil bradycardia", 1)
            except Exception:  # noqa: BLE001 - a store outage is not the subject
                pass
        assert any(row[0] == "vector_store" for row in protection.authorised), (
            "the vector store call was not authorised"
        )


class TestExternalMemoryIsBehindTheWall:
    """ADV15-9/G-8c is closed - the content written IS protected text - but the
    architecture review carried a caveat into A-4: the write was not routed
    through authorise(), so the policy row was never consulted and the
    identifier assertion never ran on this sink."""

    def test_a_write_carrying_a_removed_identifier_is_refused(self) -> None:
        from mao.memory.interface import Mem0MemoryStore

        protection = RequestProtection(trace_id="t")
        protection.record_identifier("NAME", "Harold Nkemdirim", InputChannel.QUERY)
        with protected_request(protection):
            with pytest.raises(EgressRefused):
                Mem0MemoryStore().remember(
                    "Summarise the plan for Harold Nkemdirim", "ok", "u"
                )

    def test_a_read_carrying_a_removed_identifier_is_refused(self) -> None:
        """There was no policy row for a read at b63311d, so this call could
        not have been authorised even by a call site that wanted to be."""
        from mao.memory.interface import Mem0MemoryStore

        protection = RequestProtection(trace_id="t")
        protection.record_identifier("NAME", "Harold Nkemdirim", InputChannel.QUERY)
        with protected_request(protection):
            with pytest.raises(EgressRefused):
                Mem0MemoryStore().recall("what about Harold Nkemdirim", "u")

    def test_a_clean_read_records_the_authorisation(self) -> None:
        from mao.memory.interface import Mem0MemoryStore

        protection = RequestProtection(trace_id="t")
        with protected_request(protection):
            try:
                Mem0MemoryStore().recall("donepezil", "u")
            except Exception:  # noqa: BLE001 - mem0 may be unconfigured here
                pass
        assert any(row[1] == "memory_read" for row in protection.authorised)


class TestThePolicyTableDocumentsOnlyCoverageTheCodeHas:
    """A-4's deeper finding: "Recorded: yes. Enforced: for one destination of
    seven." A row nothing reads documents coverage the code does not have, and
    nothing could tell."""

    def test_every_destination_with_a_row_has_a_production_call_site(self) -> None:
        """Deliberately excludes `egress/` itself.

        A first version of this test scanned all of `mao/` and passed, because
        `sinks.py` NAMES every destination - so a module that declares wrappers
        nobody calls satisfied it exactly as the unenforced policy rows had.
        The question is whether something OUTSIDE the egress package reaches
        each destination, so the egress package is not allowed to answer it.
        """
        from mao.trust.egress.policy import named_flows

        callers = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in pathlib.Path("mao").rglob("*.py")
            if "trust/egress/" not in path.as_posix()
        )
        wrapper_for = {
            "web_search": "authorise_web_search",
            "scholarly_api": "authorise_scholarly",
            "vector_store": "authorise_vector_",
            "external_memory": "authorise_memory_",
            "model_provider": "Destination.MODEL_PROVIDER",
        }
        unenforced = sorted(
            {
                destination.value
                for destination, _purpose, _classes in named_flows()
                if wrapper_for.get(destination.value, destination.name)
                not in callers
            }
        )
        assert unenforced == [], (
            f"the policy table declares rows for {unenforced} that no call "
            "site outside the egress package consults. A row nothing reads "
            "documents coverage the code does not have."
        )

    def test_the_wrapper_check_would_notice_a_missing_caller(self) -> None:
        """Non-vacuity for the test above: the strings it looks for are the
        ones the wrappers are actually named, so a typo in this map would make
        it fail rather than silently pass."""
        from mao.trust.egress import sinks

        for name in (
            "authorise_web_search",
            "authorise_scholarly",
            "authorise_vector_query",
            "authorise_vector_embedding",
            "authorise_memory_write",
            "authorise_memory_read",
        ):
            assert hasattr(sinks, name), f"sinks.{name} does not exist"

    def test_a_destination_with_no_rows_is_unrouted_on_purpose(self) -> None:
        from mao.trust.egress.policy import Destination, named_flows

        routed = {destination for destination, _, _ in named_flows()}
        assert Destination.MCP_TOOL not in routed
        assert Destination.ANALYTICS not in routed
