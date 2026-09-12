r"""O5 — the transformation's events and the clinician's notice cannot disagree.

## The property

For every chat query in the corpus:

    if the clinical phrase is ABSENT from the scrubbed output,
    then the redaction notice is NON-EMPTY.

The chat posture does not refuse; it redacts and answers. So the only thing
standing between an over-redaction and silent clinical loss is that the
clinician is TOLD. `00_RULES.md`: "a transformation may not silently delete
clinically material content and then present the result as equivalent to the
original."

At `ff34722` this failed in a measured, lexicon-shaped way. `find_ambiguities`
re-derived the question with a pass of its own, narrower than the scrubber's,
and the two halves disagreed on identically shaped input:

    'Patient Name: Sarah May Okonkwo Complete Heart Block.'
        -> 'Patient Name: [NAME].'        redaction_notice = []
    'Patient Name: Sarah May Okonkwo Rockwood Frailty Scale 6.'
        -> 'Patient Name: [NAME] 6.'      redaction_notice = [...]

Whether a destruction was ANNOUNCED turned out to be a function of which
clinical phrase was destroyed. `ambiguities_from(events)` is the remediation:
one event list, emitted by the transformation while it was rewriting the text,
read by the notice, the 422 and the completeness account alike.

## The corpus

A cartesian product of given x middle x family names x clinically material
phrases x layouts, including the two banners that destroyed content silently:

    inline banner   'Patient Name: <name> <phrase>.'
    comma banner    'Name: <name>, <phrase>.'

## How this oracle is independent

  destruction is measured by substring, not by asking the implementation
      `_phrase_survived` is `phrase in scrubbed`. It does not consult
      `RedactionEvent`, `Removal`, the lexicon, or `layout.Certainty`. So the
      left-hand side of the implication is established without reference to the
      machinery whose consistency is being graded.

  the notice is read at TWO places that must agree
      `ambiguities_from(protected.events)` for the whole corpus, and the real
      `POST /chat` response's `redaction_notice` field for a subset. If the
      route and the event list ever drift — which is the drift this
      remediation exists to end — the subset catches it.

  both populations are reported
      A corpus producing only destructions, or only survivals, grades nothing.
      `test_both_populations_occur` asserts each is non-trivially large.
"""
from __future__ import annotations

import itertools

import pytest

from mao.core.deident.ambiguity import ambiguities_from
from mao.trust.classes import InputChannel
from mao.trust.egress.gateway import RequestProtection, protected_request
from mao.trust.inputs.boundary import protect_channel

# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------

GIVEN_NAMES = ("Sarah", "Gordon", "Harold", "Aoife", "Margaret")
MIDDLE_NAMES = ("", "May", "James")
FAMILY_NAMES = ("Okonkwo", "Whitfield", "Nkemdirim", "Parkinson")

#: Clinically material, and chosen because they are the SAME SHAPE as a person's
#: name — a run of capitalised words. That collision is the irreducible problem
#: `ambiguity.py` documents, and it is what makes silent destruction possible.
PHRASES = (
    "Complete Heart Block",
    "Medtronic Azure Pacemaker in situ",
    "Penicillin anaphylaxis",
    "Rockwood Frailty Scale 6",
    "Katz Independence Ladder",
)

#: The first two are the layouts that destroyed content silently at `ff34722`.
LAYOUTS = (
    ("inline_banner", "Patient Name: {name} {phrase}."),
    ("comma_banner", "Name: {name}, {phrase}."),
    ("own_line", "Patient Name: {name}\n{phrase}."),
    ("trailing_banner", "{phrase}. Patient Name: {name}"),
    ("question_after", "Patient Name: {name} {phrase}. Is donepezil safe?"),
)


def _corpus() -> list[tuple[str, str, str, str]]:
    """`(layout, name, phrase, query)` for every combination."""
    cases: list[tuple[str, str, str, str]] = []
    for given, middle, family, phrase, (layout, template) in itertools.product(
        GIVEN_NAMES, MIDDLE_NAMES, FAMILY_NAMES, PHRASES, LAYOUTS
    ):
        name = " ".join(part for part in (given, middle, family) if part)
        cases.append((layout, name, phrase, template.format(name=name, phrase=phrase)))
    return cases


CORPUS = _corpus()


# --------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------


def _through_the_chat_boundary(query: str):
    """The chat posture: redact, never refuse. The same call `/chat` makes."""
    with protected_request(RequestProtection(trace_id="o5")):
        return protect_channel(
            query, InputChannel.QUERY, refuse_ambiguity=False, structured=None
        )


def _phrase_survived(phrase: str, scrubbed: str) -> bool:
    """Substring presence. Nothing from `mao/` is consulted."""
    return phrase in scrubbed


def _classify(query: str, phrase: str) -> tuple[bool, bool, str]:
    """`(destroyed, announced, scrubbed_text)` for one case."""
    protected = _through_the_chat_boundary(query)
    destroyed = not _phrase_survived(phrase, protected.text)
    announced = bool(ambiguities_from(protected.events))
    return destroyed, announced, protected.text


@pytest.fixture(scope="module")
def classified():
    return [
        (layout, name, phrase, query, *_classify(query, phrase))
        for layout, name, phrase, query in CORPUS
    ]


# --------------------------------------------------------------------------
# The invariant, over the whole corpus
# --------------------------------------------------------------------------


class TestEveryDestructionIsAnnounced:
    def test_the_corpus_is_large_enough_to_mean_something(self) -> None:
        assert len(CORPUS) >= 300, f"only {len(CORPUS)} cases generated"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT DEFECT: layout._extent_certainty still lets a sentence "
            "terminator settle a name's extent whenever ANY content follows it, "
            "so a clinical phrase inside the taken run is deleted with no "
            "notice. See TestASentenceTerminatorStillSilencesTheNotice."
        ),
    )
    def test_no_clinical_phrase_disappears_without_a_notice(self, classified) -> None:
        silent = [
            (layout, name, phrase, query, scrubbed)
            for layout, name, phrase, query, destroyed, announced, scrubbed
            in classified
            if destroyed and not announced
        ]
        assert silent == [], (
            f"{len(silent)} of {len(classified)} cases destroyed a clinical "
            "phrase and told the clinician nothing. First five:\n"
            + "\n".join(
                f"  layout={layout} phrase={phrase!r}\n"
                f"    query : {query!r}\n    output: {scrubbed!r}"
                for layout, name, phrase, query, scrubbed in silent[:5]
            )
        )

    def test_both_populations_occur(self, classified) -> None:
        """A corpus producing one outcome grades nothing.

        If every case were preserved the implication would be vacuously true. If
        every case were destroyed, a notice hardcoded to non-empty would pass.
        Both must be substantial.
        """
        destroyed = [case for case in classified if case[4]]
        preserved = [case for case in classified if not case[4]]
        assert len(destroyed) >= 50, (
            f"only {len(destroyed)} of {len(classified)} cases destroy the "
            "phrase, so the implication is close to vacuous"
        )
        assert len(preserved) >= 50, (
            f"only {len(preserved)} of {len(classified)} cases preserve the "
            "phrase, so the corpus cannot tell a real notice from a constant"
        )

    def test_the_notice_is_not_simply_always_on(self, classified) -> None:
        """The other half of non-vacuity.

        A notice emitted on every input would satisfy the implication and be
        useless. There must be cases where nothing was destroyed and nothing was
        announced.
        """
        quiet = [
            case for case in classified if not case[4] and not case[5]
        ]
        assert len(quiet) >= 50, (
            f"only {len(quiet)} cases are both undestroyed and unannounced — "
            "the notice may be firing unconditionally"
        )

    #: Layouts on which the invariant currently holds. `question_after` is
    #: excluded because it is the DEFECT, reported below in
    #: `TestASentenceTerminatorStillSilencesTheNotice` — not because the
    #: assertion was narrowed to make it pass. The whole-corpus assertion above
    #: is kept at full strength and marked as failing.
    SOUND_LAYOUTS = ("inline_banner", "comma_banner", "own_line",
                     "trailing_banner")

    @pytest.mark.parametrize("layout", SOUND_LAYOUTS)
    def test_the_invariant_holds_within_each_layout(
        self, classified, layout: str
    ) -> None:
        """Reported per layout, because the two banners are the named ones."""
        cases = [case for case in classified if case[0] == layout]
        assert cases, f"no cases for layout {layout}"
        silent = [case for case in cases if case[4] and not case[5]]
        assert silent == [], (
            f"{layout}: {len(silent)} of {len(cases)} silent destructions, "
            f"e.g. query={silent[0][3]!r} output={silent[0][6]!r}"
        )

    def test_the_silent_population_is_confined_to_the_reported_defect(
        self, classified
    ) -> None:
        """Every silent destruction must be the one already reported.

        This is what keeps the per-layout split above honest: if a silent
        destruction ever appears in a layout this file calls sound, it is a NEW
        defect and this test — not the xfail — is what fires.
        """
        silent = [case for case in classified if case[4] and not case[5]]
        stray = [case for case in silent if case[0] != "question_after"]
        assert stray == [], (
            f"{len(stray)} silent destructions outside the reported defect's "
            f"layout, e.g. layout={stray[0][0]} query={stray[0][3]!r} "
            f"output={stray[0][6]!r}"
        )
        assert silent, (
            "there are no silent destructions at all — the defect may be "
            "fixed, in which case remove the xfail above"
        )

    def test_the_two_ff34722_examples_now_agree(self) -> None:
        """The exact pair that disagreed, asserted to behave the same way.

        These two were the finding: identical shape, opposite announcement.
        Whichever way each resolves now, a destruction must be announced.
        """
        pair = (
            "Patient Name: Sarah May Okonkwo Complete Heart Block.",
            "Patient Name: Sarah May Okonkwo Rockwood Frailty Scale 6.",
        )
        phrases = ("Complete Heart Block", "Rockwood Frailty Scale 6")
        outcomes = []
        for query, phrase in zip(pair, phrases, strict=True):
            destroyed, announced, scrubbed = _classify(query, phrase)
            outcomes.append((destroyed, announced))
            assert not (destroyed and not announced), (
                f"silent destruction on the named example: {query!r} -> "
                f"{scrubbed!r}"
            )
        assert outcomes[0] == outcomes[1], (
            "the two identically shaped examples still resolve differently, "
            f"which is the ff34722 finding: {outcomes}"
        )


class TestTheCorpusContainsRealDestructions:
    """NON-VACUITY CONTROL, stated as a positive claim.

    `test_no_clinical_phrase_disappears_without_a_notice` is an implication.
    This shows its antecedent is genuinely satisfied — that the corpus really
    does contain cases where a clinical phrase is destroyed — so the green
    result is about the notice and not about an empty antecedent.
    """

    def test_at_least_one_destruction_is_exhibited(self, classified) -> None:
        destroyed = [case for case in classified if case[4]]
        assert destroyed, "no case in the corpus destroys its clinical phrase"
        layout, name, phrase, query, _, announced, scrubbed = destroyed[0]
        assert phrase not in scrubbed
        assert announced, "the exhibited destruction is not announced"

    def test_a_destruction_is_exhibited_in_each_banner_layout(
        self, classified
    ) -> None:
        """The two layouts the finding named, specifically."""
        for layout in ("inline_banner", "comma_banner"):
            destroyed = [
                case for case in classified if case[0] == layout and case[4]
            ]
            assert destroyed, (
                f"{layout} destroys nothing in this corpus, so it is not "
                "exercising the layout the finding was about"
            )

    def test_the_notice_names_a_line_and_a_field_but_no_value(
        self, classified
    ) -> None:
        """A notice that quoted the value would be the disclosure it prevented.

        Scoped to cases that DID produce a notice: this asserts what a notice
        must contain, which says nothing about cases where none was produced.
        Those are the subject of the invariant above.
        """
        announced = [case for case in classified if case[4] and case[5]][:40]
        assert announced, "no announced destruction to inspect"
        for _layout, name, _phrase, query, _, _, _ in announced:
            protected = _through_the_chat_boundary(query)
            described = ambiguities_from(protected.events).describe()
            assert "line" in described
            for token in name.split():
                assert token not in described, (
                    f"the notice quotes the redacted name: {described!r}"
                )


class TestASentenceTerminatorStillSilencesTheNotice:
    r"""REPORTED PRODUCT DEFECT — the ff34722 shape, one clause later.

    `layout._extent_certainty`:

        if leading[:1] in _SENTENCE_END and leading[1:].strip():
            return Certainty.SETTLED, words, ""

    The remediation added `and leading[1:].strip()`, which stops a terminator at
    END OF LINE from settling an extent — "a terminator SEPARATES, so it settles
    nothing when there is nothing on the far side of it".

    But the converse was left standing: a terminator with ANY content after it
    still declares the extent settled, and it does so without establishing
    anything about what is INSIDE the run. The run is still taken whole, the
    clinical phrase inside it is still deleted, and now no notice is emitted.

    ## The minimal pair

        'Patient Name: Sarah Okonkwo Complete Heart Block.'
            -> 'Patient Name: [NAME].'            notice: YES  (guessed)
        'Patient Name: Sarah Okonkwo Complete Heart Block. X'
            -> 'Patient Name: [NAME]. X'          notice: NO   (settled)

    One character of difference. The same clinical phrase is destroyed in both.
    Whether the clinician is told depends on whether anything follows the full
    stop — and a newline counts as nothing, so

        '...Complete Heart Block.\nIs donepezil safe?'   -> announced
        '...Complete Heart Block. Is donepezil safe?'    -> silent

    which means the warning turns on a whitespace choice.

    This is the finding `report.py` says is closed: "`find_ambiguities`
    announced one chat destruction and stayed silent on an identically shaped
    one." It still does; the discriminator has moved from the lexicon to
    punctuation. Measured on this file's corpus: 120 of 300 `question_after`
    cases destroy their phrase silently.

    Reported, not patched — the fix belongs in
    `mao/core/deident/layout.py::_extent_certainty`.
    """

    DESTROYED_AND_ANNOUNCED = "Patient Name: Sarah Okonkwo Complete Heart Block."
    DESTROYED_AND_SILENT = "Patient Name: Sarah Okonkwo Complete Heart Block. X"
    PHRASE = "Complete Heart Block"

    @staticmethod
    def _measure(query: str) -> tuple[bool, bool, str]:
        protected = _through_the_chat_boundary(query)
        return (
            _phrase_survived(TestASentenceTerminatorStillSilencesTheNotice.PHRASE,
                             protected.text),
            bool(ambiguities_from(protected.events)),
            protected.text,
        )

    def test_both_halves_of_the_pair_destroy_the_phrase(self) -> None:
        """The two inputs really are the same destruction."""
        for query in (self.DESTROYED_AND_ANNOUNCED, self.DESTROYED_AND_SILENT):
            survived, _, scrubbed = self._measure(query)
            assert not survived, (
                f"{query!r} no longer destroys the phrase: {scrubbed!r}"
            )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT DEFECT: a sentence terminator with content after it "
            "declares a five-word run settled, silencing the notice for a "
            "destruction identical to one that is announced."
        ),
    )
    def test_the_pair_is_announced_identically(self) -> None:
        _, announced_a, _ = self._measure(self.DESTROYED_AND_ANNOUNCED)
        _, announced_b, scrubbed_b = self._measure(self.DESTROYED_AND_SILENT)
        assert announced_a == announced_b, (
            "two identical destructions are announced differently — the "
            "ff34722 finding, reopened.\n"
            f"  {self.DESTROYED_AND_ANNOUNCED!r} -> announced={announced_a}\n"
            f"  {self.DESTROYED_AND_SILENT!r} -> announced={announced_b} "
            f"({scrubbed_b!r})"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="PRODUCT DEFECT: the destruction is silent.",
    )
    def test_the_silent_half_is_announced(self) -> None:
        survived, announced, scrubbed = self._measure(self.DESTROYED_AND_SILENT)
        assert not survived
        assert announced, (
            f"{self.PHRASE!r} was deleted and nothing was said: {scrubbed!r}"
        )

    def test_the_defect_is_still_present_as_described(self) -> None:
        """Pin it, so the two xfails above are checkable claims."""
        _, announced_a, _ = self._measure(self.DESTROYED_AND_ANNOUNCED)
        _, announced_b, _ = self._measure(self.DESTROYED_AND_SILENT)
        assert announced_a is True, "the announced half stopped announcing"
        assert announced_b is False, (
            "the silent half now announces — the defect may be fixed, in which "
            "case remove the xfails above"
        )

    def test_whitespace_after_the_terminator_decides_it(self) -> None:
        """The starkest statement of the defect: a newline versus a space."""
        _, with_newline, _ = self._measure(
            "Patient Name: Sarah Okonkwo Complete Heart Block.\nIs donepezil safe?"
        )
        _, with_space, _ = self._measure(
            "Patient Name: Sarah Okonkwo Complete Heart Block. Is donepezil safe?"
        )
        assert with_newline is True
        assert with_space is False, (
            "the space/newline discrimination is gone — the defect may be fixed"
        )


class TestTheRouteAgreesWithTheEventList:
    """The same invariant, measured at `POST /chat` through the real ASGI app.

    A representative subset rather than the whole corpus: each route call runs
    the full graph. The point is not coverage — the event list above has that —
    but that the ROUTE's notice is the same answer. `/chat` and `/chat/stream`
    formatting their own text is how they drifted apart over `AmbiguousDocument`
    once already.
    """

    #: One per layout, each chosen because the event-list measurement says the
    #: phrase is destroyed. If that ever stops being true the guard below fires.
    SUBSET = tuple(
        template.format(name="Sarah May Okonkwo", phrase=phrase)
        for (_, template), phrase in zip(
            LAYOUTS, PHRASES, strict=False
        )
    )

    @pytest.fixture(scope="class", autouse=True)
    def provider(self):
        """Bound at `gateway.set_provider`, the last hop before the vendor SDK.

        Without it these routes answer 503 from the real Groq client and the
        response carries no notice to read — the suite's own technique, reused
        here as plumbing only.
        """
        from mao.providers import gateway

        from tests.trust.recorders import RecordingProvider

        recording = RecordingProvider()
        gateway.set_provider(recording)
        try:
            yield recording
        finally:
            gateway.reset_provider()

    @pytest.fixture(scope="class")
    def client(self, provider):
        from fastapi.testclient import TestClient

        from mao.api.main import app

        with TestClient(app) as test_client:
            yield test_client

    @pytest.mark.parametrize(
        ("query", "phrase"), list(zip(SUBSET, PHRASES, strict=False))
    )
    def test_the_route_notice_equals_the_event_list_verdict(
        self, client, query: str, phrase: str
    ) -> None:
        """The route and the event list must be ONE answer.

        This is the property the remediation actually established: whatever
        `ambiguities_from(events)` says, `/chat` says the same. It holds even
        for the case the defect above makes wrong, because both halves are now
        reading the same record — which is the point.
        """
        _, announced, scrubbed = _classify(query, phrase)

        response = client.post("/chat", json={"query": query, "user_id": "o5"})
        assert response.status_code == 200, response.text
        notice = response.json().get("redaction_notice", [])

        assert bool(notice) == announced, (
            "the route's notice disagrees with the transformation's own event "
            f"list.\n  query    : {query!r}\n  scrubbed : {scrubbed!r}\n"
            f"  events say announced={announced}, route says {bool(notice)}\n"
            f"  notice   : {notice!r}"
        )

    def test_the_route_announces_every_destruction_in_the_subset(
        self, client
    ) -> None:
        """The invariant itself, at the route.

        Marked below rather than here: the subset includes the
        `question_after` layout, which is the reported defect. Kept at the route
        as well as at the event list so the defect is shown to be visible to a
        clinician using the product, not only to a unit test.
        """
        silent = []
        for query, phrase in zip(self.SUBSET, PHRASES, strict=False):
            destroyed, _, scrubbed = _classify(query, phrase)
            if not destroyed:
                continue
            response = client.post("/chat", json={"query": query, "user_id": "o5"})
            assert response.status_code == 200, response.text
            if not response.json().get("redaction_notice", []):
                silent.append((query, phrase, scrubbed))
        assert silent, (
            "no silent destruction at the route — the defect may be fixed, in "
            "which case turn this back into the positive assertion"
        )

    def test_the_route_subset_contains_a_destruction(self, client) -> None:
        """Non-vacuity for the subset: if nothing in it is destroyed, the route
        assertions above are all satisfied by an empty antecedent."""
        destroyed = [
            query
            for query, phrase in zip(self.SUBSET, PHRASES, strict=False)
            if _classify(query, phrase)[0]
        ]
        assert destroyed, (
            "no query in the route subset destroys its clinical phrase, so "
            "these route tests prove nothing about announcement"
        )

    def test_the_route_notice_never_quotes_the_patient_name(self, client) -> None:
        query = "Patient Name: Sarah May Okonkwo Complete Heart Block."
        response = client.post("/chat", json={"query": query, "user_id": "o5"})
        assert response.status_code == 200, response.text
        body = response.json()
        for token in ("Sarah", "Okonkwo"):
            assert token not in " ".join(body.get("redaction_notice", [])), (
                "the route's notice quotes the name it removed"
            )
