r"""ADV15-8 — `chat_history` was a completely unscrubbed PHI egress.

`apply_input_guardrails` was applied to `request.query` and to nothing else, so
`request.chat_history` went into `make_initial_state` verbatim at every call
site, and from `state["chat_history"]` it reached the provider in three places:

    mao/agents/router.py          _format_history(state["chat_history"][-3:])
    mao/agents/graphrag_agent.py  messages.extend(state["chat_history"][-4:])
    mao/agents/critic_agent.py    messages.extend(chat_history[-4:])

Measured over the real HTTP surface, one history turn containing a name, an MRN,
an NHS number and a date of birth delivered all four to the provider.

This is ordinary traffic, not an attack. A chat UI resends the turns it
displayed, what it displayed is what the clinician typed, and the server never
returns the de-identified query for the client to send back instead — so there
is no version of a well-behaved client that avoids it.

## Why these probes are shaped this way

The recorder is bound at `mao.providers.gateway.set_provider`, the last thing
before the vendor SDK. Patching `gateway.complete` instead would replace the
egress authorisation being tested rather than observe it, which is a recorded
defect in this project — a test written that way passes whether or not the
control exists.

Every probe asserts `recorder.calls` first. A previous memory-sink assertion in
this suite was green because it watched a path that structurally could not carry
the data; with de-identification disabled entirely it stayed green.

Both routes are asserted because they have drifted before: `/chat` answered 422
where `/chat/stream` answered a bare 500 on the same refusal.

`TestTheProbeCanFail` is the mutation control. It disables the boundary's scrub
and asserts the leak reappears. If it does not, these probes are bound to the
wrong seam and prove nothing.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from mao.core.deident.report import ScrubResult
from mao.providers import gateway
from mao.providers.llm.base import ProviderResponse
from tests.trust.recorders import RecordingProvider


class _PlausibleRecorder(RecordingProvider):
    """`RecordingProvider`, answering in the shape each caller parses.

    Still bound at `gateway.set_provider` — the same real sink, recording the
    same messages. Only the reply is smarter, because the fixed one made the
    router fail to classify, the judge fail to parse and the answer be withheld,
    which left the memory sink empty and turned a probe on it vacuous. A probe
    that cannot see the data is the failure mode this suite exists to avoid, so
    the request has to actually succeed.
    """

    def complete(self, *, model_id, messages, temperature, max_tokens):
        super().complete(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        system = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        )
        if "VERDICT" in system:
            text = "VERDICT: PASS. No concerns."
        elif "safety" in system and "groundedness" in system:
            text = '{"safety": 10, "groundedness": 9, "notes": "ok"}'
        elif "ungrounded_claims" in system:
            text = '{"ungrounded_claims": []}'
        elif "missing" in system:
            text = '{"missing": []}'
        else:
            # Doubles as a router label and as prose: `_classify` reads the
            # first recognised token, everything else reads the sentence.
            text = "graphrag: donepezil is a cholinesterase inhibitor."
        return ProviderResponse(text=text, input_tokens=10, output_tokens=10)

# Synthetic. Invented for this fixture.
PATIENT_NAME = "Harold Nkemdirim"
MRN = "RGT/44219/B"
NHS_NUMBER = "943 476 5919"
DOB = "12/03/1948"

PHI = [PATIENT_NAME, MRN, NHS_NUMBER, DOB]

# What a chat UI resends: the turn the clinician typed, exactly as displayed.
HISTORY_TURN = (
    f"Patient Name: {PATIENT_NAME}, MRN: {MRN}, "
    f"NHS Number: {NHS_NUMBER}, DOB: {DOB}. "
    "He was started on donepezil 5 mg at the memory clinic."
)

CHAT_HISTORY = [
    {"role": "user", "content": HISTORY_TURN},
    {"role": "assistant", "content": "Donepezil is a cholinesterase inhibitor."},
]

FOLLOW_UP = "Should the dose be titrated to 10 mg given his bradycardia?"


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch):
    """Everything the process would have put on the wire, at the real sink."""
    from mao.agents import clinical_agent, graphrag_agent
    from mao.memory.interface import reset_memory_store, set_memory_store
    from mao.safety import verification
    from tests.trust.recorders import RecordingMemory

    provider = _PlausibleRecorder()
    gateway.set_provider(provider)
    memory = RecordingMemory()
    set_memory_store(memory)
    provider.memory = memory

    # Only the hops that leave the machine are replaced. Every node the request
    # passes through runs its production code.
    monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
    monkeypatch.setattr(graphrag_agent, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(graphrag_agent, "_web_search", lambda *a, **k: [])
    monkeypatch.setattr(graphrag_agent, "_live_pubmed_search", None, raising=False)
    monkeypatch.setattr(verification, "check_all_claims", lambda claims, premise: [])

    yield provider

    gateway.reset_provider()
    reset_memory_store()


@pytest.fixture
def client(recorder):
    from mao.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _post_chat(client: TestClient):
    return client.post(
        "/chat",
        json={
            "query": FOLLOW_UP,
            "user_id": "history-probe",
            "chat_history": CHAT_HISTORY,
        },
    )


def _post_stream(client: TestClient):
    return client.post(
        "/chat/stream",
        json={
            "query": FOLLOW_UP,
            "user_id": "history-probe",
            "chat_history": CHAT_HISTORY,
        },
    )


class TestHistoryIsDeIdentifiedBeforeItLeavesTheProcess:
    def test_chat_returns_an_answer(self, client, recorder) -> None:
        assert _post_chat(client).status_code == 200

    def test_chat_stream_returns_an_answer(self, client, recorder) -> None:
        assert _post_stream(client).status_code == 200

    def test_no_identifier_from_the_history_reaches_the_provider_on_chat(
        self, client, recorder
    ) -> None:
        _post_chat(client)
        assert recorder.calls, "the provider was never called — the probe is vacuous"
        assert recorder.leaked(PHI) == [], (
            "these identifiers from chat_history were sent to the provider"
        )

    def test_no_identifier_from_the_history_reaches_the_provider_on_stream(
        self, client, recorder
    ) -> None:
        _post_stream(client)
        assert recorder.calls, "the provider was never called — the probe is vacuous"
        assert recorder.leaked(PHI) == [], (
            "these identifiers from chat_history were sent to the provider"
        )

    def test_the_history_still_reaches_the_provider(self, client, recorder) -> None:
        """Otherwise the assertions above pass because nothing was sent at all.

        De-identification must remove the identifiers and leave the clinical
        content, or the model answers a follow-up question having been told
        nothing about the conversation it follows.
        """
        _post_chat(client)
        assert recorder.calls, "the provider was never called — the probe is vacuous"
        assert "donepezil" in recorder.text()

    def test_no_identifier_reaches_the_memory_store(self, client, recorder) -> None:
        """A second, PERSISTENT sink on the same path.

        `remember()` receives the query and the response, so this asserts the
        query channel rather than the history one — both are minted by the same
        boundary call, and a regression that reopened one would reopen both.
        """
        _post_chat(client)
        assert recorder.memory.calls, "nothing was written to memory — probe is vacuous"
        assert recorder.memory.leaked(PHI) == []


class TestTheProbeCanFail:
    """Mutation control — disable the boundary's scrub, expect RED.

    If these still pass, the probes above are bound to a seam that cannot see
    the defect and their green is worth nothing.
    """

    @pytest.fixture
    def unprotected(self, monkeypatch: pytest.MonkeyPatch):
        from mao.trust.inputs import boundary

        monkeypatch.setattr(
            boundary, "scrub_with_report", lambda text: ScrubResult(text=text)
        )

    def test_chat_leaks_every_identifier_without_the_boundary(
        self, client, recorder, unprotected
    ) -> None:
        _post_chat(client)
        assert recorder.calls, "the provider was never called — the probe is vacuous"
        assert sorted(recorder.leaked(PHI)) == sorted(PHI), (
            "disabling the scrub did not reopen the leak, so the probe is not "
            "bound to the seam that carries it"
        )

    def test_chat_stream_leaks_every_identifier_without_the_boundary(
        self, client, recorder, unprotected
    ) -> None:
        _post_stream(client)
        assert recorder.calls, "the provider was never called — the probe is vacuous"
        assert sorted(recorder.leaked(PHI)) == sorted(PHI)


class TestNeitherRouteCanBuildStateFromRawText:
    """The structural half: one boundary means one constructor.

    `make_initial_state` takes plain strings and is still right for a CLI demo
    or a unit test, where there is no request and no patient. A route reaching
    for it is the defect itself — that is literally the line ADV15-8 was.
    """

    def test_the_chat_module_uses_only_the_protected_constructor(self) -> None:
        import ast
        import inspect

        from mao.api import main

        tree = ast.parse(inspect.getsource(main))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "initial_state_from_protected" in called
        assert "make_initial_state" not in called, (
            "a chat route built graph state from strings instead of from the "
            "protected input boundary's output"
        )

    def test_the_boundary_is_entered_on_both_routes(self) -> None:
        import inspect

        from mao.api import main

        for route in (main.chat, main.chat_stream_endpoint):
            source = inspect.getsource(route)
            assert "protect_chat_request(" in source, route.__name__
            assert "protected_request(" in source, route.__name__


class TestTheStreamedAnswerStillArrives:
    def test_the_stream_carries_tokens_and_a_metadata_frame(
        self, client, recorder
    ) -> None:
        """Guards against a refusal that looks like success.

        The protection is bound with a context manager that the route exits
        before the response body is iterated, so a mistake there shows up as an
        empty or truncated stream rather than as an exception.
        """
        body = _post_stream(client).text
        assert "data: " in body
        assert body.rstrip().endswith("data: [DONE]")
        meta = [
            line for line in body.splitlines() if line.startswith("data: __meta__:")
        ]
        assert meta, f"no metadata frame in the stream: {body[:200]!r}"
        json.loads(meta[0][len("data: __meta__:") :])
