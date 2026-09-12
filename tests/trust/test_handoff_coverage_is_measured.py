r"""A-1, ADV16-6, AR17-1 and ADV17-2: the Safe Handoff's loss guard, three times.

Four reviews have now found this guard unable to see the loss it exists to
detect, by a different mechanism each time:

    A-1      `extract.py` read
                 if _DOSE.search(stripped):        ...; matched = True
                 elif _is_clinical_line(stripped): ...; matched = True
                 if not matched and _is_clinical_line(stripped): uncovered.append(...)
             If `_is_clinical_line` is true the `elif` already ran, so the final
             guard is unsatisfiable and `uncovered` was unconditionally empty.

    ADV16-6  the denominator consulted the SAME lexicon as the de-identifier,
             so a clinical line that lexicon does not recognise was in NEITHER
             the numerator NOR the denominator - not "uncovered" but invisible.

    AR17-1   the denominator became a line count and a new predicate excluded
    ADV17-2  lines BEFORE they reached it. `_carries_nothing_to_lose` examined
             only the value half of the first colon, so
             `Complete Heart Block confirmed, discussed with next of kin: [NAME]`
             was excluded from both sides. Measured: four clinical facts
             including an implanted pacemaker dropped from the projection handed
             to a model asked whether a bradycardic drug was safe, at coverage
             1.0000, `uncertainties` empty, no refusal and no notice.
             Second route, same predicate: `\[[A-Z_]+\]` matched a clinician's
             own `Pacing mode: [DDDR]`.

`00_RULES.md`: *"If repeated fixes to one mechanism repeatedly introduce new
failures in the same invariant class, stop patching symptoms and escalate the
abstraction/design before another remediation round."*

So this file no longer grades a predicate. It grades an ACCOUNT: every segment
of the source is attributed to exactly one of four states, and completeness is
what the account says rather than what a rule concluded. The tests are written
against the states, not against the arithmetic, because the arithmetic was never
the part that failed.

They run through the REAL protected-input boundary wherever the boundary's own
redaction record is what the account depends on. Hand-written `[NAME]` text is
not what a document looks like after the boundary has run, and grading the
accounting on hand-written placeholders is how the previous version of this file
could pass while the real path lost a contraindication.
"""
from __future__ import annotations

import dataclasses

import pytest

from mao.trust.classes import InputChannel
from mao.trust.egress.gateway import RequestProtection, protected_request
from mao.trust.handoff.accounting import (
    Segment,
    SegmentState,
    SourceAccounting,
)
from mao.trust.handoff.compiler import (
    MINIMUM_COVERAGE,
    HandoffRefused,
    compile_handoff,
)
from mao.trust.handoff.extract import extract
from mao.trust.inputs.boundary import protect_channel

LETTER = """Clinic letter.
The patient was reviewed in the memory clinic today.
Medtronic Azure Pacemaker in situ, Attain Performa Lead.
Bristol Stool Chart type 6 throughout.
Continuing Healthcare Funding application submitted.
Aspree Extension Trial participant, arm two.
Donepezil 10 mg once daily.
Blood pressure 128/76.
Is donepezil safe for this patient?
"""

PACEMAKER = "Medtronic Azure Pacemaker in situ, Attain Performa Lead."

#: The AR17-1 / ADV17-2 document, as a clinician writes it. Every clinical
#: statement sits on the LABEL side of a colon whose value is an identifier the
#: boundary removes correctly, which is the shape that was invisible.
ATTRIBUTION_LETTER = """Patient Name: Harold James Nkemdirim
MRN: RGT/44219/B
Medtronic Azure Pacemaker in situ, lead checked by: Dr Aoife Rankin
Complete Heart Block confirmed, discussed with next of kin: Sarah Okonkwo
Penicillin anaphylaxis, alert added by: Dr John Smith
Warfarin INR 4.8, phoned through to: 0113 496 0231
Is donepezil safe for this patient?
"""


def through_the_boundary(raw: str, question: str, **structured: str):
    """The real protected-input boundary, then the real compiler.

    The boundary's own redaction events are what let the account tell an
    identifier it removed from clinical content it did not carry. Constructing
    the protected text by hand and omitting them is not a shortcut to the same
    thing: it is a different, more pessimistic account, and grading against it
    would let the real path regress unseen.
    """
    with protected_request(RequestProtection(trace_id="test")):
        protected = protect_channel(
            raw,
            InputChannel.REPORT,
            refuse_ambiguity=False,
            structured=dict(structured),
        )
    handoff = compile_handoff(
        protected.text,
        question=question,
        structured=dict(structured),
        events=tuple(protected.events),
    )
    return protected, handoff


class TestEverySegmentIsAccountedForSomewhere:
    """The property that replaces "is this line in the denominator".

    Exhaustiveness is the whole point: there is no branch in which a segment is
    dropped from both sides, because there is no branch in which a segment is
    dropped at all.
    """

    def test_the_pacemaker_line_is_accounted_somewhere(self) -> None:
        facts = extract(LETTER)
        accounted = facts.findings + facts.unresolved_text()
        assert any(PACEMAKER in item for item in accounted), (
            "the line is in no state at all, so no measurement could see it"
        )

    def test_no_meaning_bearing_segment_is_missing_from_the_account(self) -> None:
        """Stated as a total, not as a spot check.

        Every line of the source that carries a word must appear in exactly one
        segment. This is the assertion the three previous mechanisms would each
        have failed, and it cannot be satisfied by adding a word to a list.
        """
        facts = extract(LETTER)
        accounted_lines = {
            segment.line
            for segment in facts.accounting.segments
            if segment.state is not SegmentState.STRUCTURAL
            or segment.text.strip()
        }
        expected = {
            index
            for index, line in enumerate(LETTER.splitlines())
            if any(character.isalnum() for character in line)
        }
        assert expected <= accounted_lines

    def test_coverage_is_below_one_when_content_was_not_carried(self) -> None:
        facts = extract(LETTER)
        assert facts.coverage() < 1.0, (
            f"coverage() reported {facts.coverage()} on a document that lost "
            "content. That is the ADV16-6 mechanism: 1.00 precisely when the "
            "loss is total."
        )

    def test_the_lost_lines_are_named_not_merely_counted(self) -> None:
        facts = extract(LETTER)
        assert facts.completeness().unresolved_lines, "nothing is reported as lost"

    def test_unresolved_is_reachable_at_all(self) -> None:
        """A-1: the branch that populated the equivalent was unsatisfiable."""
        facts = extract("qwertyuiop asdfghjkl\nzxcvbnm poiuytrewq\n")
        assert facts.accounting.in_state(SegmentState.UNRESOLVED)

    def test_the_account_does_not_consult_the_lexicon(self) -> None:
        """Two documents with the same shape and the same number of typed facts
        must score the same, whatever vocabulary they happen to use."""
        known = extract("Donepezil 10 mg once daily.\nWibble wobble flim flam.\n")
        unknown = extract("Donepezil 10 mg once daily.\nQwerty uiop asdf ghjk.\n")
        assert known.coverage() == unknown.coverage()


class TestTheAttributionLineIsNotInvisible:
    """AR17-1 and ADV17-2, at the boundary that produced them.

    `<clinical statement>: <identifier>` is ordinary documentation practice —
    "discussed with", "referred to", "phoned through to", "alert added by" — and
    the boundary redacting the name at the end of it is the boundary working
    correctly. The measured harm was that the MORE correctly the boundary
    redacted, the more clinical lines disappeared from the measurement.
    """

    def test_no_clinical_statement_is_both_absent_and_unreported(self) -> None:
        """The disjunction that actually matters.

        For every clinically material phrase: it is either IN the payload, or
        the payload says the projection is incomplete and names the line it came
        from. Never both absent and complete. This is weaker than "carry
        everything" on purpose — Phase 1 does not claim to understand arbitrary
        clinical prose — and it is exactly the claim `00_RULES.md` makes
        unconditional.
        """
        _, handoff = through_the_boundary(
            ATTRIBUTION_LETTER, "Is donepezil safe for this patient?"
        )
        rendered = handoff.synthesis_context.render()
        report = handoff.facts.completeness()
        for phrase in ("Pacemaker", "Heart Block", "Penicillin", "Warfarin"):
            if phrase.lower() in rendered.lower():
                continue
            assert not report.complete, (
                f"{phrase!r} is absent from the payload and the projection "
                "still reports itself complete — the ADV17-2 signature"
            )
            assert report.unresolved_lines, (
                f"{phrase!r} is absent and no source line is named"
            )

    def test_the_attribution_lines_are_not_excluded_from_the_account(self) -> None:
        """The mechanism, stated directly: they must be accountable at all."""
        protected, handoff = through_the_boundary(
            ATTRIBUTION_LETTER, "Is donepezil safe for this patient?"
        )
        report = handoff.facts.completeness()
        assert report.accountable >= 4, (
            f"only {report.accountable} segment(s) are accountable in a letter "
            "with four clinical statements; the attribution clause is still "
            "removing them from both sides"
        )

    def test_a_clinician_typed_bracketed_token_is_not_read_as_a_redaction(
        self,
    ) -> None:
        r"""The second route into the same exclusion.

        `_PLACEHOLDER = r'\[[A-Z_]+\]'` matched any bracketed uppercase token,
        so a pacing mode, a rhythm and an NYHA class were each accounted as an
        identifier this system had removed and their lines vanished. The account
        reads redaction EVENTS, and no event was recorded for text a clinician
        typed, so there is nothing here to widen.
        """
        for line in ("Pacing mode: [DDDR]", "Rhythm: [AAI]", "Functional class: [NYHA]"):
            report = extract(line).completeness()
            assert report.accountable == 1, (
                f"{line!r} was excluded from the account entirely"
            )


class TestARedactedIdentifierIsNotCountedAsLoss:
    """The trap the architecture review predicted in its A-1 remediation note:
    "with the catch-all removed, every clinical line currently carried verbatim
    into findings becomes uncovered, coverage collapses, and the 50% threshold
    starts refusing ordinary documents."

    The account avoids it WITHOUT an exclusion. An identifier the boundary
    removed is `IDENTIFIER_REMOVED` and the label that introduced it is
    `STRUCTURAL`; neither is in the numerator or the denominator because each is
    in its own state. That is the difference between accounting for something
    and excluding it, and it is why there is nothing here for a fourth predicate
    to get wrong.
    """

    REPORT = (
        "MEMORY CLINIC REPORT\n"
        "Patient Name: Harold Nkemdirim\n"
        "MRN: A1234567\n"
        "Address: 14 Beckett Street, Leeds\n"
        "DOB: 14/03/1941\n"
        "Contact: harold.nkemdirim@nhs.net\n"
        "Findings: moderate hippocampal atrophy, MMSE 21/30.\n"
    )

    def test_the_identifier_lines_are_not_counted_as_clinical_loss(self) -> None:
        """The A-1 trap, stated as what it actually is.

        The trap is that an ordinary letterhead's identifier lines get counted
        as content the projection failed to carry, collapsing coverage to 0.286
        and refusing an answerable document. The account prevents that by
        putting each removal and each attributing label in its OWN state, and
        that is what is asserted here — not a particular coverage figure, which
        would be tuning the assertion to the corpus.
        """
        _, handoff = through_the_boundary(self.REPORT, "Summarise this report")
        report = handoff.facts.completeness()
        assert report.identifiers_removed >= 5, (
            "the boundary's removals are not reaching the account at all"
        )
        assert report.structural >= 5, (
            "the labels that introduced those removals are being counted as "
            "clinical content"
        )
        assert report.coverage() > 0.286, (
            f"the letterhead scored {report.coverage()}, at or below the 0.286 "
            "the A-1 remediation note predicted for the collapsed measurement"
        )

    def test_a_residual_the_scrubber_left_is_reported_rather_than_hidden(
        self,
    ) -> None:
        r"""The account is PESSIMISTIC where the transformation was incomplete,
        and that is the safe direction.

        Two lines of this letterhead are unresolved, and both honestly:

          `Address: 14 Beckett Street, Leeds`  redacts to
          `Address: [ADDRESS], Leeds` — the town survived the address rule, so
          a place name really is still there and really is not carried.

          `Contact: harold.nkemdirim@nhs.net` — `Contact` is not a field name
          the scrubber's label grammar knows, so nothing attributes the email
          to it and the word is left accountable.

        Neither is a false negative. The first surfaces a genuine residual and
        the second costs a little coverage on a document that is not refused. A
        rule that suppressed either would have to decide that some leftover text
        is not worth mentioning, and that decision is the one this subsystem has
        got wrong four times.
        """
        _, handoff = through_the_boundary(self.REPORT, "Summarise this report")
        report = handoff.facts.completeness()
        assert not report.complete
        assert report.unresolved_lines, "an unresolved segment names no line"
        for text in handoff.facts.unresolved_text():
            assert text.strip(), "an empty segment is being counted as a loss"

    def test_and_is_therefore_not_refused(self) -> None:
        _, handoff = through_the_boundary(self.REPORT, "Summarise this report")
        assert handoff.synthesis_context is not None

    def test_the_clinical_line_is_still_carried(self) -> None:
        """Non-vacuity: coverage is 1.0 because the content WAS carried, not
        because the document was scored as having none."""
        _, handoff = through_the_boundary(self.REPORT, "Summarise this report")
        report = handoff.facts.completeness()
        assert report.carried >= 1
        assert report.identifiers_removed >= 4, (
            "the boundary's own removals are not reaching the account"
        )
        assert any(
            "hippocampal atrophy" in finding
            for finding in handoff.facts.findings
        )

    def test_a_line_with_a_redaction_AND_content_still_counts(self) -> None:
        """A clinical line that happens to contain a redaction is accountable.

        This is the case the value-half predicate got wrong in both directions.
        """
        _, handoff = through_the_boundary(
            "Seen with Dr Aoife Rankin for Medtronic Azure Pacemaker check.\n",
            "Is the device working?",
        )
        assert handoff.facts.completeness().accountable == 1


class TestCoverageIsOneOnlyWhenThereWasNothingToLose:
    def test_an_empty_document_scores_one(self) -> None:
        assert extract("").coverage() == 1.0

    def test_a_whitespace_document_scores_one(self) -> None:
        assert extract("\n\n   \n").coverage() == 1.0

    def test_a_document_whose_lines_all_became_facts_scores_one(self) -> None:
        facts = extract("Donepezil 10 mg once daily.\nBlood pressure 128/76.\n")
        assert facts.coverage() == 1.0
        assert facts.completeness().complete
        assert facts.completeness().unresolved == 0


class TestTheAccountIsNonVacuous:
    """00_RULES prefers a mutation test for any claim that a guard would detect
    a bypass. Three mutations, each re-creating one of the three historical
    defects, and each of which the assertions above must notice.
    """

    def test_forcing_every_segment_carried_restores_the_defect(self) -> None:
        """b63311d / ff34722: nothing is ever reported as lost."""
        facts = extract(LETTER)
        assert facts.coverage() < 1.0
        as_it_was = dataclasses.replace(
            facts,
            accounting=SourceAccounting(
                segments=tuple(
                    dataclasses.replace(segment, state=SegmentState.TYPED_FACT)
                    if segment.state is SegmentState.UNRESOLVED
                    else segment
                    for segment in facts.accounting.segments
                )
            ),
        )
        assert as_it_was.coverage() == 1.0
        assert as_it_was.completeness().complete, (
            "with every segment marked carried the account must report complete "
            "again — if it does not, these tests are not measuring what they "
            "claim to"
        )

    def test_forcing_every_segment_excluded_restores_the_other_defect(self) -> None:
        """AR17-1: the loss is invisible because the segment is not there.

        An account with no accountable segments reports coverage 1.0 and
        complete. That is correct for a document that genuinely had nothing, and
        it is the exact reading the exclusion predicate produced for a document
        that had four contraindications — which is why the assertions above are
        written against `accountable`, not only against `coverage`.
        """
        facts = extract(LETTER)
        emptied = dataclasses.replace(facts, accounting=SourceAccounting())
        assert emptied.coverage() == 1.0
        assert emptied.completeness().complete
        assert emptied.completeness().accountable == 0

    def test_an_unresolved_segment_makes_the_projection_incomplete(self) -> None:
        """The hard invariant, isolated from any document."""
        one = SourceAccounting(
            segments=(
                Segment(0, 0, 5, SegmentState.TYPED_FACT, "fact", "findings"),
                Segment(1, 6, 12, SegmentState.UNRESOLVED, "residue"),
            )
        )
        assert not one.report().complete
        assert one.report().unresolved_lines == (2,)
        none = SourceAccounting(
            segments=(Segment(0, 0, 5, SegmentState.TYPED_FACT, "fact", "findings"),)
        )
        assert none.report().complete


class TestTheThinProjectionRefusalCanFire:
    """A-1 consequence 1: `_require_substance` compared a constant 1.0 to 0.5,
    so the entire HandoffRefused 'thin projection' branch was unreachable and
    only the `has_clinical_facts()` 'empty' branch was live."""

    UNREADABLE = "Donepezil 10 mg once daily.\n" + "\n".join(
        f"qwertyuiop{n} asdfghjkl{n} zxcvbnm{n}" for n in range(10)
    )

    def test_a_document_the_extractor_mostly_cannot_read_is_refused(self) -> None:
        with pytest.raises(HandoffRefused):
            compile_handoff(self.UNREADABLE, question="Is donepezil safe?")

    def test_the_refusal_names_the_fields_that_would_resolve_it(self) -> None:
        with pytest.raises(HandoffRefused) as raised:
            compile_handoff(self.UNREADABLE, question="Is donepezil safe?")
        assert raised.value.wanted, "a refusal with no remedy is a wall"
        assert "structured patient fields" in str(raised.value)

    def test_an_ordinary_document_is_not_refused(self) -> None:
        """Non-vacuity. A guard that refuses everything is not a guard."""
        handoff = compile_handoff(
            "Donepezil 10 mg once daily.\nBlood pressure 128/76.\n",
            question="Is donepezil safe?",
        )
        assert handoff.synthesis_context is not None

    def test_the_threshold_is_a_recorded_judgement(self) -> None:
        """Pinned so a later reader does not mistake it for a measurement, and
        so nobody concludes that the threshold is what closes the class. Both
        `ff34722` reviews defeated the threshold argument the same way: where
        the loss was invisible, coverage was 1.00, so no value would have
        refused. `TestTheProjectionNeverClaimsToBeComplete` is the control that
        does not depend on a number."""
        assert MINIMUM_COVERAGE == 0.5


class TestTheProjectionNeverClaimsToBeComplete:
    """A-1 consequence 2, and AR17-2's correction of the first attempt at it.

    `providers/gateway.py` tells the model that "facts that could not be carried
    safely are absent rather than summarised... Where a fact you would need is
    missing, say which fact and do not assume it." At b63311d the signal was
    structurally always empty, so the model was told to look for something never
    sent. At ff34722 the signal became the RESIDUE ITSELF — every line no
    grammar could parse, rendered verbatim to the model, cached in Redis and
    written to a trace. Measured: a full personal name and a contact extension
    in the payload.

    Measuring the loss and transmitting the residue are different requirements.
    """

    def test_the_payload_always_states_its_completeness(self) -> None:
        _, handoff = through_the_boundary(LETTER, "Is donepezil safe?")
        rendered = handoff.synthesis_context.render()
        assert "Case completeness:" in rendered

    def test_an_incomplete_projection_says_so_in_words(self) -> None:
        _, handoff = through_the_boundary(LETTER, "Is donepezil safe?")
        assert not handoff.facts.completeness().complete
        rendered = handoff.synthesis_context.render()
        assert "INCOMPLETE" in rendered
        assert "do not assume the omitted content was immaterial" in rendered

    def test_the_residue_itself_never_reaches_the_payload(self) -> None:
        """AR17-2. The account travels; the text does not."""
        _, handoff = through_the_boundary(
            LETTER + "Dictated by Sarah Thompson, medical secretary, ext 4471.\n",
            "Is donepezil safe?",
        )
        rendered = handoff.synthesis_context.render()
        residue = handoff.facts.unresolved_text()
        assert residue, "nothing was unresolved, so this test grades nothing"
        for piece in residue:
            assert piece.strip() not in rendered, (
                "unparsed note prose reached the external synthesis payload"
            )
        assert handoff.synthesis_context.uncertainties == (), (
            "`uncertainties` is carrying the residue again"
        )

    def test_the_completeness_report_carries_no_source_text(self) -> None:
        """Stated as a property of the type, not of one document."""
        _, handoff = through_the_boundary(LETTER, "Is donepezil safe?")
        report = handoff.facts.completeness()
        for value in dataclasses.asdict(report).values():
            assert not isinstance(value, str), (
                f"CompletenessReport carries a string field: {value!r}"
            )

    def test_a_complete_projection_says_that_instead(self) -> None:
        """Non-vacuity: the statement is not a constant warning."""
        _, handoff = through_the_boundary(
            "Donepezil 10 mg once daily.\nBlood pressure 128/76.\n",
            "Is donepezil safe?",
        )
        rendered = handoff.synthesis_context.render()
        assert "Complete:" in rendered
        assert "INCOMPLETE" not in rendered
