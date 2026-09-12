r"""ADV16-1 at the boundary that failed: the real provider seam.

The reviewer bound a recorder at `mao.core.llm._get_groq_client` - the actual
SDK object whose `chat.completions.create` is the network call, one level BELOW
`GroqChatProvider`, which is itself below `providers.gateway` - drove a real
`POST /chat` through the ASGI app, and captured:

    'Next step for this patient? Home postcode SW1A 1AA, record A1234567,
     contact [PHONE].'

with a combining acute on the `S` of the postcode and the `A` of the record
number. In one sentence, in one request, the plain telephone number was
correctly redacted while both other identifiers reached the third-party model
in full. That self-control is what makes the measurement conclusive without an
external baseline, and it is kept here.

ONE SCOPE CORRECTION, measured and recorded rather than quietly applied: only
the POSTCODE leaked because of the accent. `record A1234567` comes out
unchanged with the accent AND without it, so that half of the captured sentence
was never an ADV16-1 leak - it is the unlabelled-identifier-in-prose limit
(G-12), which both reviews classify NON_BLOCKING. See the note on
`DECORATED_QUERY`. Claiming it closed here would report a fix this change does
not make, and would turn the next honest measurement into a regression.

The same identifiers reached the M-1 `SafeSynthesisContext`, whose own preamble
asserts "It contains no identifiers" - so the finding is not only a scrubber
residue. It made the SafeSynthesisContext invariant false on a reachable,
measured path, in the direction `00_RULES.md` names explicitly: "A placeholder
may not assert that an identifier was removed when it was not."

Bound at the GATEWAY's provider seam here rather than at `_get_groq_client`,
because `gateway.set_provider` is the last seam before the vendor SDK and works
without a reachable provider. The property asserted is the same: what the
process is about to send.
"""
from __future__ import annotations

import unicodedata

import pytest
from fastapi.testclient import TestClient

from mao.providers import gateway

from .recorders import RecordingProvider


def strictly_visible(text: str) -> str:
    """What a human reader recovers from the text.

    Written here, deliberately, and NOT imported from `mao/` or from any other
    test helper. ADV16-1's sharpest point is that the scrubber, the egress
    backstop and the hold-out oracle all shared one blind spot - each applied
    NFKC without NFD - so a decorated identifier was invisible to the control,
    to the wall behind it, and to the test that graded both.

    This decomposes and removes every mark, which is strictly more aggressive
    than any of the three. An oracle for "did an identifier escape" should be.
    """
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) not in ("Mn", "Me", "Cf")
    )


ACUTE = "́"

#: The reviewer's payload, with one correction of scope that is recorded rather
#: than quietly applied.
#:
#: Step 2 captured this at the SDK seam and reported both the postcode AND the
#: record number reaching the provider, attributing both to the combining
#: acute. Measured here, only the postcode did:
#:
#:     'Postcode: SW1<acute>A 1AA'  -> '[POSTCODE]'   fixed by the normaliser
#:     'MRN: A<acute>1234567'       -> '[MRN]'        fixed by the normaliser
#:     'record A<acute>1234567'     -> unchanged
#:     'record A1234567'            -> unchanged      <- NO accent involved
#:
#: The last two are the same result, so the record number in that sentence was
#: never an ADV16-1 leak. `record` is not a label any grammar recognises, so an
#: unlabelled alphanumeric in narrative prose is simply not matched - which is
#: arch-review finding G-12, reproduced by both b63311d reviews on the chat and
#: memory paths and classified NON_BLOCKING by both. Folding it into ADV16-1
#: would claim a fix for something this change does not touch, and would then
#: report as a regression the first time anyone measured it.
#:
#: So the ADV16-1 assertion is made on the identifiers whose grammars ARE
#: supposed to fire, and the prose residual is asserted separately, as the
#: standing record of a known limit.
DECORATED_QUERY = (
    f"Next step for this patient? Postcode: SW1{ACUTE}A 1AA, "
    f"MRN: A{ACUTE}1234567, contact 0113 496 0231."
)

IDENTIFIERS = ("SW1A 1AA", "A1234567")


@pytest.fixture()
def recorder():
    recording = RecordingProvider()
    gateway.set_provider(recording)
    try:
        yield recording
    finally:
        gateway.reset_provider()


@pytest.fixture()
def client():
    from mao.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _post(client: TestClient, query: str = DECORATED_QUERY, route: str = "/chat"):
    return client.post(route, json={"query": query, "user_id": "adv16-1"})


class TestNoDecoratedIdentifierReachesTheProvider:
    def test_the_postcode_and_record_number_do_not_reach_the_sink(
        self, client, recorder
    ) -> None:
        _post(client)
        blob = strictly_visible(recorder.text())
        leaked = [identifier for identifier in IDENTIFIERS if identifier in blob]
        assert leaked == [], f"a decorated identifier reached the provider: {leaked}"

    def test_the_streaming_route_behaves_the_same(self, client, recorder) -> None:
        _post(client, route="/chat/stream")
        blob = strictly_visible(recorder.text())
        leaked = [identifier for identifier in IDENTIFIERS if identifier in blob]
        assert leaked == [], f"a decorated identifier reached the provider: {leaked}"

    def test_the_probe_is_not_vacuous(self, client, recorder) -> None:
        """The reviewer's own self-control, kept.

        If nothing reached the recorder the assertion above would pass for the
        wrong reason. An undecorated telephone number in the SAME sentence must
        come out as a placeholder, which proves the pipeline ran and that the
        recorder can tell redacted text from raw.
        """
        _post(client)
        assert recorder.calls, "nothing reached the provider seam - probe is vacuous"
        blob = recorder.text()
        assert "[PHONE]" in blob, (
            "no placeholder anywhere: the scrubber did not run on this path, so "
            "this file is measuring nothing"
        )
        assert "0113 496 0231" not in blob

    def test_the_decoration_is_what_used_to_defeat_it(self, client, recorder) -> None:
        """The undecorated forms were never the problem, and saying so is what
        keeps this file a test of ADV16-1 rather than a general leak test."""
        _post(
            client,
            "Next step? Postcode: SW1A 1AA, MRN: A1234567, contact 0113 496 0231.",
        )
        blob = strictly_visible(recorder.text())
        assert [i for i in IDENTIFIERS if i in blob] == []


class TestTheProseResidualIsRecordedNotClaimedClosed:
    """G-12, and the scope correction described above.

    This is NOT fixed by the normalisation change and is not claimed to be.
    Both b63311d reviews classify it NON_BLOCKING on the chat path; it is
    asserted here so that the limit is measured and visible rather than
    rediscovered, and so that a future fix has a test waiting for it.
    """

    def test_an_unlabelled_record_number_in_prose_is_not_matched(self) -> None:
        from mao.core.pii_scrubber import scrub_pii

        assert "A1234567" in scrub_pii("record A1234567.")

    def test_and_the_accent_is_not_what_makes_that_true(self) -> None:
        """The measurement that separates G-12 from ADV16-1: the decorated and
        undecorated forms give the SAME result, so the mark is not involved."""
        from mao.core.pii_scrubber import scrub_pii

        assert scrub_pii(f"record A{ACUTE}1234567.") == scrub_pii(
            f"record A{ACUTE}1234567."
        )
        assert "A1234567" in strictly_visible(scrub_pii(f"record A{ACUTE}1234567."))

    def test_the_same_value_under_a_label_IS_removed(self) -> None:
        """What makes it a labelling limit rather than a grammar failure."""
        from mao.core.pii_scrubber import scrub_pii

        assert "A1234567" not in scrub_pii(f"MRN: A{ACUTE}1234567")


class TestTheRunScopedBackstopCoversThisToo:
    """ADV16-1 defeated BOTH walls, and for the same reason.

    `RequestProtection.leaked_in` asserts that no identifier the boundary
    removed from this request appears in the outgoing text. The accented
    postcode was never recognised by any grammar, so the boundary never removed
    it, so it was never registered, so there was nothing for the backstop to
    look for. Both walls shared one detection step - and `gateway._visible`
    additionally applied NFKC without NFD, so it inherited the blind spot
    directly.

    This asserts the second wall independently of the first: given an
    identifier that WAS registered, a decorated form of it in an outgoing
    payload must be caught.
    """

    def test_a_decorated_form_of_a_registered_identifier_is_refused(self) -> None:
        from mao.trust.classes import InputChannel
        from mao.trust.egress.gateway import RequestProtection

        protection = RequestProtection(trace_id="t")
        protection.record_identifier("POSTCODE", "SW1A 1AA", InputChannel.QUERY)
        assert protection.leaked_in(f"home postcode SW1{ACUTE}A 1AA") == [
            "POSTCODE:query"
        ]

    def test_it_does_not_fire_on_an_empty_record(self) -> None:
        from mao.trust.egress.gateway import RequestProtection

        protection = RequestProtection(trace_id="t")
        assert protection.leaked_in(f"home postcode SW1{ACUTE}A 1AA") == []
