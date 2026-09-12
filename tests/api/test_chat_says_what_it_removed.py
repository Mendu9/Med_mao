r"""A-2, at the route, which is the boundary that failed.

`protect()` called `find_ambiguities` for exactly two channels, REPORT and
TRANSCRIPT. QUERY and CHAT_HISTORY never consulted it at all, so the
`refuse_ambiguity=False` the chat route passes was not even the operative
control for them - there was no check to disable, and the scrubber's guess was
taken unconditionally. Measured over the independent hold-out corpus: 227 of
1476 cases lose clinical content on QUERY and 227 on HISTORY, across 38
distinct phrases, six of which are implanted cardiac devices and leads.

The boundary justified the posture as "the clinician wrote the text and reads
the answer, so an over-redaction is visible and recoverable while a leak is
not". The judgement is sound and is not disturbed here. The PREMISE was not:
`ChatResponse` carried `response`, `agent_used`, `intent`, `metadata`,
`request_id`, `latency_ms`, `sources` and `web_sources`, none of them the
protected text, and `api/protected_input.py` states as a design premise that
"the server never returns the de-identified query to the client". The clinician
saw an answer, not the question the model was asked.

So the boundary was never choosing between a leak and an over-redaction. The
third option - the one `00_RULES.md` prescribes, and the one the upload path
already takes - is to say what was removed.

Why the existing hold-out assertion did not catch it: it asserted that
`find_ambiguities` WOULD return a report, which is a property of the DETECTOR.
All 227 destroying cases satisfied that, so the suite was green while the
reachable chat path refused nothing and signalled nothing. These assertions are
on the channel and on the route.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

#: The architecture review's worked example, verbatim from its probe.
DESTROYING_QUERY = "Patient Name: Sarah Parkinson    Katz Independence Ladder"
LOST_PHRASE = "Katz Independence Ladder"
ORDINARY_QUERY = "Is donepezil safe in bradycardia?"


@pytest.fixture()
def client():
    from mao.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _post(client: TestClient, query: str, history=None, route: str = "/chat"):
    body = {"query": query, "user_id": "a2-probe"}
    if history:
        body["chat_history"] = history
    return client.post(route, json=body)


class TestTheBoundaryConsultsTheDetectorOnTheseChannels:
    """Bound at the boundary rather than the route, so a failure here says the
    detector is not being called at all rather than that a field is missing."""

    def _protect(self, query: str, history=()):
        from mao.trust.classes import RawSensitiveInput
        from mao.trust.inputs.boundary import protect

        return protect(
            RawSensitiveInput(query=query, chat_history=tuple(history)),
            trace_id="t",
            refuse_ambiguity=False,
        )

    def test_the_query_channel_reports_an_ambiguity(self) -> None:
        assert self._protect(DESTROYING_QUERY).chat_ambiguities, (
            "find_ambiguities is still never called on the QUERY channel"
        )

    def test_the_history_channel_reports_one_too(self) -> None:
        protected = self._protect(
            ORDINARY_QUERY, [("user", DESTROYING_QUERY)]
        )
        assert protected.chat_ambiguities

    def test_the_clinical_phrase_really_is_destroyed(self) -> None:
        """Non-vacuity. If nothing were lost on this input the assertions above
        would be about a case that does not exist."""
        assert LOST_PHRASE not in self._protect(DESTROYING_QUERY).query.text

    def test_an_ordinary_query_reports_nothing(self) -> None:
        """A notice on every request tells the caller nothing."""
        assert not self._protect(ORDINARY_QUERY).chat_ambiguities

    def test_the_report_channel_is_unaffected(self) -> None:
        """`ambiguities` keeps its existing meaning so nothing downstream that
        reads it changes behaviour."""
        assert self._protect(DESTROYING_QUERY).ambiguities.items == []


class TestTheChatRouteSaysWhatItRemoved:
    def test_a_destroying_query_returns_a_redaction_notice(self, client) -> None:
        body = _post(client, DESTROYING_QUERY).json()
        assert body.get("redaction_notice"), (
            "clinical content was removed from the query and the caller was "
            "told nothing"
        )

    def test_the_notice_names_the_line_and_the_field(self, client) -> None:
        notice = " ".join(_post(client, DESTROYING_QUERY).json()["redaction_notice"])
        assert "line" in notice.lower()
        assert "field" in notice.lower()

    def test_the_notice_never_quotes_the_identifier(self, client) -> None:
        """It is returned over HTTP and written to a log. A notice that quoted
        the identifier would be the disclosure the redaction just prevented."""
        notice = " ".join(_post(client, DESTROYING_QUERY).json()["redaction_notice"])
        for token in ("Sarah", "Parkinson"):
            assert token not in notice, f"the notice quoted {token!r}"

    def test_an_ordinary_query_gets_no_notice(self, client) -> None:
        assert not _post(client, ORDINARY_QUERY).json().get("redaction_notice")

    def test_history_is_covered_too_and_not_only_the_query(self, client) -> None:
        body = _post(
            client,
            ORDINARY_QUERY,
            history=[{"role": "user", "content": DESTROYING_QUERY}],
        ).json()
        assert body.get("redaction_notice")

    def test_the_answer_itself_does_not_carry_the_identifier(self, client) -> None:
        """The leak direction is unchanged and must stay that way: 0 of 1476
        hold-out cases leak on this path."""
        blob = _post(client, DESTROYING_QUERY).text
        assert "Sarah Parkinson" not in blob


class TestTheStreamingRouteSaysTheSameThing:
    """ADV15-2 / G-3 was these two routes drifting apart. They must not again,
    and both now read the notice from one formatter."""

    def test_the_stream_carries_the_notice(self, client) -> None:
        response = _post(client, DESTROYING_QUERY, route="/chat/stream")
        assert response.status_code == 200
        assert "redaction_notice" in response.text

    def test_the_stream_notice_quotes_no_identifier(self, client) -> None:
        body = _post(client, DESTROYING_QUERY, route="/chat/stream").text
        assert "Sarah Parkinson" not in body

    def test_an_ordinary_query_streams_an_empty_notice(self, client) -> None:
        body = _post(client, ORDINARY_QUERY, route="/chat/stream").text
        assert '"redaction_notice": []' in body or "redaction_notice" in body
