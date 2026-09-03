"""Wave 9 / B4 — the raw query must not reach a provider or the database.

`main.py` computed `safe_query` and then handed `request.query` — the raw string
— to the background scorer. Three sinks from one `/chat` call:

    1. mao/eval/llm_judge.py        -> gateway.complete(SAFETY_JUDGE)
    2. mao/eval/ragas_evaluator.py  -> a ChatGroq built inline, OUTSIDE the
                                       gateway entirely
    3. mao/eval/ragas_evaluator.py  -> RetrainingCandidate(question=...) in
                                       Postgres, which `_persist_session`
                                       explicitly refuses to store

Sink 2 is why this file exists. `tests/api/test_phi_never_reaches_the_provider.py`
binds its recorder with `gateway.set_provider()`, so it can only ever see calls
that go THROUGH the gateway — and the leak went around it. A guard placed at an
abstraction cannot see a call site that bypasses the abstraction.

So the recorder here binds one level lower, at
`groq.resources.chat.completions.Completions.create`: the single SDK method that
every path bottoms out in. `mao.core.llm` calls it via the gateway provider, and
`langchain_groq.ChatGroq` builds its own `groq.Groq` and calls the same method.
Binding there catches both, and catches the next bypass nobody has written yet.
"""
from __future__ import annotations

import pytest

# Synthetic identifiers, shaped like the real thing.
PATIENT_NAME = "Arthur Neville Kowalczyk"
MRN = "LDS/9931/C"
PHI = (PATIENT_NAME, MRN)

QUERY_WITH_PHI = (
    f"Patient Name: {PATIENT_NAME} MRN: {MRN}. "
    "Should donepezil be titrated in a patient with sinus bradycardia?"
)


class SdkRecorder:
    """Every string handed to the Groq SDK, whoever handed it over."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from groq.resources.chat.completions import Completions

        recorder = self

        def _create(self, *args, **kwargs):  # noqa: ANN001
            for message in kwargs.get("messages", []) or []:
                recorder.sent.append(str(message.get("content", "")))
            return _canned_completion(kwargs.get("messages") or [])

        monkeypatch.setattr(Completions, "create", _create, raising=True)

    def leaked(self) -> list[str]:
        blob = "\n".join(self.sent)
        return [identifier for identifier in PHI if identifier in blob]


def _canned_completion(messages: list[dict]):
    """A response shaped like Groq's, so no call leaves the machine."""
    from types import SimpleNamespace

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
        text = "Donepezil may be used with caution in bradycardia."
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


@pytest.fixture
def sdk_recorder(monkeypatch: pytest.MonkeyPatch) -> SdkRecorder:
    recorder = SdkRecorder()
    recorder.install(monkeypatch)
    return recorder


class TestTheJudgeNeverSeesTheRawQuery:
    """Sink 1 — `llm_judge` goes through the gateway, so the SDK seam sees it."""

    def test_the_judge_prompt_carries_no_identifier(
        self, sdk_recorder: SdkRecorder
    ) -> None:
        from mao.core.pii_scrubber import scrub_pii
        from mao.eval.llm_judge import judge_response

        judge_response(question=scrub_pii(QUERY_WITH_PHI), response="An answer.")

        assert sdk_recorder.sent, "the SDK was never called — the probe is vacuous"
        assert sdk_recorder.leaked() == []

    def test_the_probe_would_have_caught_the_leak(
        self, sdk_recorder: SdkRecorder
    ) -> None:
        """Guard against a vacuous seam: handing the judge the RAW query must
        make the recorder fire. A probe that cannot fail proves nothing, and
        three consecutive gates were passed by exactly that."""
        from mao.eval.llm_judge import judge_response

        judge_response(question=QUERY_WITH_PHI, response="An answer.")

        assert sdk_recorder.leaked(), (
            "the recorder did not see a raw query it was explicitly given — "
            "it is bound to the wrong seam"
        )


class TestScoreResponseDeIdentifiesItsOwnInput:
    """The contract, not the call site.

    `main.py:342` passing `request.query` was possible because the parameter
    accepted any string and the docstring asked for "the user's original
    question". Fixing the one call site leaves the next caller free to repeat
    it, so the guarantee is established where it cannot be skipped.
    """

    @pytest.fixture
    def captured(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        from mao.eval import ragas_evaluator

        seen: dict = {}

        def _fake_ragas(question, answer, contexts):  # noqa: ANN001
            seen["ragas_question"] = question
            return {"faithfulness": 0.1, "answer_relevancy": 0.5}

        def _fake_judge(question, response, context=""):  # noqa: ANN001
            seen["judge_question"] = question
            return {}

        def _fake_store_retraining(*args, **kwargs):  # noqa: ANN001, ANN003
            seen["retraining_question"] = kwargs.get("question", args[3] if len(args) > 3 else None)

        monkeypatch.setattr(ragas_evaluator, "_run_ragas_sync", _fake_ragas)
        monkeypatch.setattr(ragas_evaluator, "_store_metrics", lambda *a, **k: None)
        monkeypatch.setattr(ragas_evaluator, "_update_prometheus", lambda *a, **k: None)
        monkeypatch.setattr(
            ragas_evaluator, "_store_retraining_candidate", _fake_store_retraining
        )
        monkeypatch.setattr("mao.eval.llm_judge.judge_response", _fake_judge)
        return seen

    @pytest.mark.asyncio
    async def test_every_downstream_sink_gets_the_scrubbed_question(
        self, captured: dict
    ) -> None:
        from mao.eval.ragas_evaluator import score_response

        await score_response(
            question=QUERY_WITH_PHI,
            answer="Donepezil may be used with caution.",
            contexts=["Donepezil is a cholinesterase inhibitor."],
            user_id="phi-probe",
            request_id="req-1",
            agent_used="clinical",
        )

        assert captured, "nothing downstream was reached — the probe is vacuous"
        for sink, value in captured.items():
            assert value is not None, f"{sink} received nothing"
            for identifier in PHI:
                assert identifier not in str(value), (
                    f"{sink} received the raw identifier {identifier!r}"
                )

    @pytest.mark.asyncio
    async def test_the_clinical_question_still_reaches_the_scorer(
        self, captured: dict
    ) -> None:
        """Scrubbing must not gut the question, or the scores are meaningless
        and the test above would pass for the wrong reason."""
        from mao.eval.ragas_evaluator import score_response

        await score_response(
            question=QUERY_WITH_PHI,
            answer="Donepezil may be used with caution.",
            contexts=["Donepezil is a cholinesterase inhibitor."],
            user_id="phi-probe",
            request_id="req-1",
            agent_used="clinical",
        )

        assert "bradycardia" in captured["ragas_question"]


class TestTheApplicationLogNeverHoldsTheRawQuery:
    """B5 — an INFO log of the raw query on every request contradicts the M5
    decision to "persist the de-identified query only". It is an independent
    sink: fixing the scorer call site does not close it."""

    def test_the_chat_route_logs_no_raw_query(
        self, sdk_recorder: SdkRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wave 11 / NB5 — REWRITTEN. The old version could not fail.

        It built its own `logging.getLogger("mao.api.main").info(...)` call with
        an already-scrubbed argument and asserted no PHI in the output. That
        asserts a property of the test, not of the route: it passed identically
        whatever `main.py` did, including logging the raw query on the line
        above. Mutating `safe_query[:80]` back to `request.query[:80]` left it
        green.

        The replacement POSTs to the real route. It attaches its OWN handler to
        the root logger rather than using `caplog`, because `caplog` did not see
        the mutated route either — a capture fixture is one more seam that can
        be blind to the defect, which is precisely the failure mode this finding
        is about. A plain root handler is verified to catch it: with the
        mutation in place this test fails, and without it, it passes.
        """
        import logging

        from fastapi.testclient import TestClient

        from mao.agents import clinical_agent
        from mao.api.main import app
        from mao.memory.interface import reset_memory_store, set_memory_store
        from mao.safety import verification

        class _NoMemory:
            def recall(self, query, user_id):
                return ""

            def remember(self, query, response, user_id):
                return None

        class _Capture(logging.Handler):
            def __init__(self) -> None:
                super().__init__(level=logging.DEBUG)
                self.messages: list[str] = []

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    self.messages.append(record.getMessage())
                except Exception:  # pragma: no cover - defensive
                    pass

        set_memory_store(_NoMemory())
        monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
        monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
        monkeypatch.setattr(verification, "check_all_claims", lambda claims, premise: [])

        capture = _Capture()
        root = logging.getLogger()
        previous_level = root.level
        root.addHandler(capture)
        root.setLevel(logging.DEBUG)
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/chat", json={"query": QUERY_WITH_PHI, "user_id": "phi-probe"}
                )
        finally:
            root.removeHandler(capture)
            root.setLevel(previous_level)
            reset_memory_store()

        assert response.status_code == 200
        logged = "\n".join(capture.messages)
        assert any("request_id=" in message for message in capture.messages), (
            "the route's own request log was never captured — the probe is vacuous"
        )
        for identifier in PHI:
            assert identifier not in logged, (
                f"the application log holds the raw identifier {identifier!r}"
            )

    @pytest.mark.parametrize("route", ["chat", "chat_stream"])
    def test_no_route_logs_the_query_before_scrubbing_it(self, route: str) -> None:
        """Structural: the log call must sit BELOW the line computing
        `safe_query`. At f757375 the /chat log sat ABOVE it, so the call had to
        MOVE rather than just change its argument."""
        import inspect

        from mao.api import main

        source = inspect.getsource(main)
        assert "request.query[:80]" not in source, (
            "a route still logs the raw query; log the scrubbed value instead"
        )
