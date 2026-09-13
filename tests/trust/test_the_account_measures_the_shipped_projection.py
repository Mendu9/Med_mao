r"""ADV19-1: the account and the payload must measure the SAME object.

## The defect this closes

`SourceAccounting` decided `TYPED_FACT` from the EXTRACTOR's intermediate facts.
`compile_handoff` then built the shipped `SafeSynthesisContext` from
CALLER-SUPPLIED structured overrides — `findings`, `medications` and the related
effective fields — so a segment could stay in the numerator of `coverage()` after
its represented value had been displaced from the payload that actually ships.

Measured at `ea46e7e` through the real `protect_channel(..., InputChannel.REPORT)`
and the real `compile_handoff`, differing ONLY by the presence of a structured
key:

    CONTROL no patient_fields            ATTACK findings= + medications=
    coverage 1.0000  complete True       coverage 1.0000  complete True
    Complete heart block    IN payload   Complete heart block    ABSENT
    Penicillin anaphylaxis  IN payload   Penicillin anaphylaxis  ABSENT
    Warfarin 5mg od         IN payload   Warfarin 5mg od         ABSENT

    "Case completeness: Complete: all 3 clinically accountable source
     segment(s) are represented in this projection."

All three were absent. `mao/providers/gateway.py` instructs the model to read
that line first, and the question asked was whether donepezil — a bradycardic
drug — is safe.

## Why this file does not grade a predicate

This is the FIFTH failure in the absent-and-complete class, and the first that
is not an exclusion predicate inside the account. The four before it
(`_is_clinical_line`, `_carries_nothing_to_lose`, `_PLACEHOLDER`,
`_MARKDOWN_HEADING`) were each a rule that decided what to leave out. This one
is a DISAGREEMENT: the account measured the extractor's output and the payload
was built from the caller's. A sixth predicate cannot reach it, and neither can
a merge rule — merging narrows the population without making the two objects
agree.

So the property asserted here is the one that makes `accounting.py`'s
`TYPED_FACT` contract true BY CONSTRUCTION rather than by assumption:

    A source segment may be TYPED_FACT only if its represented meaning is
    present in the FINAL effective `SafeSynthesisContext` that actually ships,
    and completeness is derived from that same final projection.

Every assertion below interrogates the RENDERED payload — the bytes the external
model receives — and never the extractor's intermediate view. An oracle that
asked the account about itself would share a detection step with the mechanism
it grades, which is the `00_RULES` Wave 13 rule and the shape of AR19-1.
"""
from __future__ import annotations

import itertools
import re

import pytest

from mao.trust.classes import InputChannel
from mao.trust.egress.gateway import RequestProtection, protected_request
from mao.trust.handoff.accounting import SegmentState
from mao.trust.handoff.compiler import HandoffRefused, compile_handoff
from mao.trust.inputs.boundary import protect_channel

#: The decisive document. Every line is a contraindication to the drug the
#: question asks about, which is what makes silent displacement a safety defect
#: rather than a reporting one.
DECISIVE_LETTER = (
    "Patient Name: Harold Nkemdirim\n"
    "MRN: RGT/44219/B\n"
    "Complete heart block\n"
    "Penicillin allergy - anaphylaxis\n"
    "Warfarin 5mg od\n"
)
QUESTION = "Is donepezil safe for this patient?"

#: The same case with vitals attached, and the reason it exists.
#:
#: On `DECISIVE_LETTER` an override displaces so much of the accountable content
#: that coverage falls under `MINIMUM_COVERAGE` and the compiler REFUSES. A
#: refusal satisfies every property in this file trivially — it makes no
#: completeness claim, so it cannot make a false one — which means a suite built
#: only on that letter would pass by refusing and would never once exercise the
#: path that actually ships a payload.
#:
#: Vitals cannot be displaced by any structured key, so they hold coverage above
#: the threshold while the findings and medications are displaced. Every
#: override shape below therefore PROCESSES, renders a payload, reports itself
#: INCOMPLETE and names the displaced lines — which is the outcome both ea46e7e
#: reviewers asked for, and the one worth asserting.
VITALS_LETTER = (
    "Patient Name: Harold Nkemdirim\n"
    "MRN: RGT/44219/B\n"
    "Complete heart block\n"
    "Penicillin allergy - anaphylaxis\n"
    "Warfarin 5mg od\n"
    "BP 128/76\n"
    "HR 36\n"
    "Temperature 36.8\n"
)

#: The two keys that displace extracted clinical content, and the two the
#: product's own refusal message instructs the caller to send.
FINDINGS_OVERRIDE = {"findings": "mild cognitive impairment"}
MEDICATIONS_OVERRIDE = {"medications": "donepezil 10mg od"}
BOTH_OVERRIDE = {**FINDINGS_OVERRIDE, **MEDICATIONS_OVERRIDE}


def through_the_boundary(raw: str, question: str, **structured: str):
    """The real upload posture, then the real compiler.

    The boundary's own redaction events are what let the account tell an
    identifier it removed from clinical content it did not carry. Building the
    protected text by hand and omitting them grades a different, more
    pessimistic account, and the real path could regress unseen behind it.
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
    return handoff


#: The age bands, restated here rather than imported.
#:
#: AR19-1 is the finding that an oracle sharing a detection step with the
#: mechanism it guards cannot fail: the shipped provenance oracle skipped exactly
#: the segments its implementation excluded, because both spelled
#: "meaning-bearing" as `str.isalnum()`. So this file derives "can the payload be
#: asked to show this segment?" from its OWN definitions. If the implementation's
#: banding changes, this table stops agreeing with it and these tests fail, which
#: is the direction a contract oracle should fail in.
_BANDS = ((90, "90_or_over"), (75, "older_adult_75_89"), (65, "older_adult_65_74"), (18, "adult"))


def _band_for(age: int) -> str:
    for floor, name in _BANDS:
        if age >= floor:
            return name
    return "under_18"


def _payload_can_show(segment, context) -> bool:
    """Whether the SHIPPED projection can be asked to show this segment.

    Defined per field by this file, because the represented meaning of a segment
    is not always its source text: a carried age ships as a BAND and a carried
    vital ships as a number. An oracle that only compared source text would
    report a false violation on every agreeing age, and one that asked the
    implementation what it represents would be asking the mechanism to grade
    itself.
    """
    text = segment.text.strip()
    if not text:
        return True
    field = segment.carried_as
    if field == "age_group":
        digits = re.search(r"\d{1,3}", text)
        return digits is not None and context.age_group == _band_for(int(digits.group()))
    if field == "vitals":
        return any(value in text for value in context.vitals.values())
    if field == "labs":
        return any(value in text for value in context.labs.values())
    # findings, medications, and anything a future extractor adds: the segment
    # is the clinician's own line and the payload carries it verbatim or not
    # at all. An unknown field falls here and fails closed.
    return text.lower() in context.render().lower()


def displaced_typed_facts(handoff) -> tuple:
    """TYPED_FACT segments the FINAL rendered payload cannot be asked to show.

    This is `accounting.py`'s TYPED_FACT contract read literally — *"this
    segment became a field of the projection, and the projection can be asked to
    show it"* — and tested against the object that ships rather than against the
    account's own opinion of itself.
    """
    return tuple(
        segment
        for segment in handoff.facts.accounting.in_state(SegmentState.TYPED_FACT)
        if not _payload_can_show(segment, handoff.synthesis_context)
    )


class TestEveryTypedFactIsRepresentedByTheShippedProjection:
    """The contract `accounting.py` states, tested against what actually ships.

    Not "the account is self-consistent" — the account was perfectly
    self-consistent while three contraindications went missing. The question is
    whether the payload can be asked to show what the account says it carries.
    """

    @pytest.mark.parametrize(
        "structured",
        [
            pytest.param({}, id="control-no-override"),
            pytest.param(FINDINGS_OVERRIDE, id="findings-override"),
            pytest.param(MEDICATIONS_OVERRIDE, id="medications-override"),
            pytest.param(BOTH_OVERRIDE, id="both-overrides"),
        ],
    )
    def test_no_typed_fact_segment_is_absent_from_the_payload(
        self, structured: dict[str, str]
    ) -> None:
        try:
            handoff = through_the_boundary(DECISIVE_LETTER, QUESTION, **structured)
        except HandoffRefused:
            # A refusal makes no completeness claim, so it cannot make a false
            # one. That is a stronger outcome than processing, not a weaker one.
            return
        displaced = displaced_typed_facts(handoff)
        assert not displaced, (
            "these segments are counted TYPED_FACT and the projection cannot be "
            "asked to show them: "
            + "; ".join(
                f"line {segment.line + 1} carried_as={segment.carried_as} "
                f"{segment.text.strip()!r}"
                for segment in displaced
            )
        )

    def test_the_isolating_control_still_carries_everything(self) -> None:
        """Without an override the document's own content must still ship.

        The fix must not buy its correctness by dropping content that was
        genuinely carried. This is the control that makes every other assertion
        in the file mean something.
        """
        handoff = through_the_boundary(DECISIVE_LETTER, QUESTION)
        payload = handoff.synthesis_context.render().lower()
        for phrase in ("complete heart block", "penicillin", "warfarin"):
            assert phrase in payload, f"{phrase!r} lost from the un-overridden payload"
        assert handoff.facts.completeness().complete


class TestDisplacedFactsCannotCoexistWithComplete:
    """The absent-and-complete property, through the two harmful keys.

    This is the invariant five gates have now failed: a clinically material
    phrase absent from the payload while the payload affirmatively states that
    every accountable segment is represented.
    """

    @pytest.mark.parametrize(
        "structured",
        [
            pytest.param(FINDINGS_OVERRIDE, id="findings-override"),
            pytest.param(MEDICATIONS_OVERRIDE, id="medications-override"),
            pytest.param(BOTH_OVERRIDE, id="both-overrides"),
        ],
    )
    def test_an_override_that_displaces_content_is_never_reported_complete(
        self, structured: dict[str, str]
    ) -> None:
        try:
            handoff = through_the_boundary(DECISIVE_LETTER, QUESTION, **structured)
        except HandoffRefused:
            return
        report = handoff.facts.completeness()
        payload = handoff.synthesis_context.render().lower()
        absent = [
            phrase
            for phrase in ("complete heart block", "penicillin", "warfarin")
            if phrase not in payload
        ]
        if absent:
            assert not report.complete, (
                f"{absent} absent from the payload at coverage "
                f"{report.coverage():.4f} while it states: "
                f"{report.describe()!r}"
            )

    def test_the_rendered_statement_itself_does_not_claim_completeness(self) -> None:
        """The statement is what the model is told to read first.

        Asserted on the rendered bytes rather than on `report.complete`, because
        the payload is what travels and a statement that disagreed with the flag
        would be the same defect one layer out.
        """
        try:
            handoff = through_the_boundary(DECISIVE_LETTER, QUESTION, **BOTH_OVERRIDE)
        except HandoffRefused:
            return
        payload = handoff.synthesis_context.render()
        if "complete heart block" not in payload.lower():
            assert "Case completeness: Complete:" not in payload, (
                "the payload states Complete while the document's own heart "
                f"block finding is absent from it:\n{payload}"
            )


class TestDisplacedLinesAreNamedInTheIncompletenessMetadata:
    """A clinician told something is missing must be pointed at the right line.

    `test_clinical_loss_is_accounted.py` states the requirement in its own
    words: *"A clinician told 'something is missing' but pointed at the wrong
    line cannot act on it."* At `ea46e7e` a document losing four facts on four
    lines reported `INCOMPLETE: 1 of 6 ... (source line(s) 5)`.
    """

    @pytest.mark.parametrize(
        "structured",
        [
            pytest.param(FINDINGS_OVERRIDE, id="findings-override"),
            pytest.param(MEDICATIONS_OVERRIDE, id="medications-override"),
            pytest.param(BOTH_OVERRIDE, id="both-overrides"),
        ],
    )
    def test_every_displaced_source_line_appears_in_unresolved_lines(
        self, structured: dict[str, str]
    ) -> None:
        try:
            handoff = through_the_boundary(DECISIVE_LETTER, QUESTION, **structured)
        except HandoffRefused:
            return
        report = handoff.facts.completeness()
        payload = handoff.synthesis_context.render().lower()

        # Ground truth derived OUTSIDE the account: which source lines carry a
        # clinical phrase the payload does not show.
        lines = DECISIVE_LETTER.splitlines()
        missing = {
            number
            for number, line in enumerate(lines, start=1)
            for phrase in ("Complete heart block", "Penicillin allergy", "Warfarin 5mg")
            if phrase in line and phrase.lower() not in payload
        }
        assert missing <= set(report.unresolved_lines), (
            f"source line(s) {sorted(missing - set(report.unresolved_lines))} carry "
            f"content absent from the payload and are not named. Reported: "
            f"{report.describe()!r}"
        )

    def test_the_unresolved_count_is_not_an_undercount(self) -> None:
        """ADV19-5: the count and the line list both derive from the account.

        A remediation that suppressed the "Complete" claim without making the
        account agree with the projection would still report `1 of 6` where four
        facts are absent. This asserts the count covers them.
        """
        try:
            handoff = through_the_boundary(DECISIVE_LETTER, QUESTION, **BOTH_OVERRIDE)
        except HandoffRefused:
            return
        report = handoff.facts.completeness()
        payload = handoff.synthesis_context.render().lower()
        absent = [
            phrase
            for phrase in ("complete heart block", "penicillin", "warfarin")
            if phrase not in payload
        ]
        assert report.unresolved >= len(absent), (
            f"{len(absent)} clinical phrases are absent but only "
            f"{report.unresolved} segment(s) are reported unresolved"
        )


class TestTheOverrideRemainsAnApprovedMechanism:
    """Structured fields must still WIN, and must still leak nothing.

    Removing the override is not the fix. `00_RULES` names structured fields as
    the remedy for ambiguity and the refusal pathway depends on them, so a
    remediation that disabled them would defeat the workflow the product's own
    422 instructs.
    """

    #: A document the override ADDS to rather than displaces: its clinical
    #: content is VITALS ONLY, and no structured key can displace those. Chosen
    #: so these assertions are UNCONDITIONAL. A `pytest.skip` here would be the
    #: W14-8 shape — an assertion that silently stops being made — inside the
    #: file written to close a finding about assertions that stopped being made.
    #:
    #: Deliberately free of any line the extractor reads as a FINDING: a
    #: `Ward round` header is one, so a findings override would displace it and
    #: this would stop being an additive case without saying so.
    ADDITIVE = "BP 128/76\nHR 72\nTemperature 36.8\n"

    def test_a_structured_finding_reaches_the_payload(self) -> None:
        handoff = through_the_boundary(
            self.ADDITIVE, QUESTION, findings="mild cognitive impairment"
        )
        assert "mild cognitive impairment" in handoff.synthesis_context.render()
        assert handoff.synthesis_context.clinically_relevant_findings == (
            "mild cognitive impairment",
        )

    def test_a_structured_medication_reaches_the_payload(self) -> None:
        handoff = through_the_boundary(
            self.ADDITIVE, QUESTION, medications="donepezil 10mg od"
        )
        assert "donepezil 10mg od" in handoff.synthesis_context.render()
        assert handoff.synthesis_context.medications_and_doses == ("donepezil 10mg od",)

    def test_an_additive_override_costs_the_document_nothing(self) -> None:
        """CONTROL. Supplying a key that displaces nothing must not pessimise.

        The remediation must react to DISPLACEMENT, not to the presence of a
        structured key. A rule that punished the key itself would refuse the
        workflow the product's own 422 instructs, which is the outcome the
        reviews explicitly forbade.
        """
        plain = through_the_boundary(self.ADDITIVE, QUESTION)
        added = through_the_boundary(
            self.ADDITIVE, QUESTION, findings="mild cognitive impairment"
        )
        assert plain.facts.completeness().complete
        assert added.facts.completeness().complete
        assert (
            added.facts.completeness().carried == plain.facts.completeness().carried
        ), "an additive override changed the account of the document's own content"

    def test_the_override_carries_no_identifier_into_the_payload(self) -> None:
        """The LEAK direction stays closed while the loss direction is fixed.

        AR18-2 is the recorded finding that caller structured fields reach the
        payload unscrubbed; it is DEFERRED and not remediated here. What this
        asserts is narrower and is a Phase 1 obligation: making the account
        agree with the projection must not open a new route for the DOCUMENT's
        protected content — the residue and the removed identifiers — to travel.
        """
        try:
            handoff = through_the_boundary(DECISIVE_LETTER, QUESTION, **BOTH_OVERRIDE)
        except HandoffRefused:
            return
        payload = handoff.synthesis_context.render()
        for identifier in ("Harold", "Nkemdirim", "RGT/44219/B"):
            assert identifier not in payload, (
                f"{identifier!r} reached the external payload"
            )
        for residue in handoff.facts.unresolved_text():
            assert residue not in payload, (
                f"residue {residue!r} reached the external payload — the account "
                "must MEASURE the loss without TRANSMITTING it"
            )

    def test_the_completeness_report_still_carries_no_source_text(self) -> None:
        """AR17-2's closure is a property of the TYPE and must stay one."""
        try:
            handoff = through_the_boundary(DECISIVE_LETTER, QUESTION, **BOTH_OVERRIDE)
        except HandoffRefused:
            return
        report = handoff.facts.completeness()
        for name, value in vars(report).items():
            assert not isinstance(value, str), (
                f"CompletenessReport.{name} is a string field; the report is the "
                "only thing that crosses the boundary and it may carry counts "
                "and line numbers only"
            )


class TestNoNaiveMergeOfCallerAndExtractedValues:
    """A merge rule is the sixth predicate, and it manufactures conflicts.

    Concatenating the caller's findings with the extractor's would narrow the
    population without making the account agree with the projection, and it
    would present a clinician's correction and the text it corrects as two
    coexisting facts about one patient.
    """

    def test_an_overridden_field_carries_only_the_callers_values(self) -> None:
        try:
            handoff = through_the_boundary(
                DECISIVE_LETTER, QUESTION, **FINDINGS_OVERRIDE
            )
        except HandoffRefused:
            return
        assert handoff.synthesis_context.clinically_relevant_findings == (
            "mild cognitive impairment",
        )

    def test_an_overridden_medication_list_carries_only_the_callers_values(
        self,
    ) -> None:
        try:
            handoff = through_the_boundary(
                DECISIVE_LETTER, QUESTION, **MEDICATIONS_OVERRIDE
            )
        except HandoffRefused:
            return
        assert handoff.synthesis_context.medications_and_doses == ("donepezil 10mg od",)

    @pytest.mark.parametrize(
        "structured",
        [
            pytest.param({}, id="control-no-override"),
            pytest.param(FINDINGS_OVERRIDE, id="findings-override"),
            pytest.param(MEDICATIONS_OVERRIDE, id="medications-override"),
            pytest.param(BOTH_OVERRIDE, id="both-overrides"),
        ],
    )
    def test_no_projection_field_contains_a_duplicate(
        self, structured: dict[str, str]
    ) -> None:
        try:
            handoff = through_the_boundary(DECISIVE_LETTER, QUESTION, **structured)
        except HandoffRefused:
            return
        context = handoff.synthesis_context
        for name in (
            "conditions",
            "medications_and_doses",
            "allergies",
            "clinically_relevant_findings",
            "uncertainties",
        ):
            values = getattr(context, name)
            assert len(values) == len(set(values)), (
                f"{name} contains a duplicate: {values}"
            )


#: Six clinically material phrases x three real medications x four override
#: shapes. The population is a breadth measurement, and the no-override column
#: is the ISOLATING CONTROL: the only difference between it and the others is
#: the presence of a caller structured key.
BREADTH_PHRASES = (
    "Complete heart block",
    "Permanent pacemaker in situ",
    "Penicillin allergy - anaphylaxis",
    "Severe aortic stenosis",
    "Sick sinus syndrome documented",
    "Second degree AV block Mobitz II",
)
BREADTH_MEDICATIONS = ("Warfarin 5mg od", "Bisoprolol 2.5mg od", "Digoxin 125mcg od")
BREADTH_OVERRIDES = (
    ({}, "none"),
    (FINDINGS_OVERRIDE, "findings"),
    (MEDICATIONS_OVERRIDE, "medications"),
    (BOTH_OVERRIDE, "both"),
)


class TestBreadthWithAnIsolatingControl:
    """The same property across a generated population, not one letter."""

    @pytest.mark.parametrize(
        ("phrase", "medication", "override"),
        [
            pytest.param(phrase, medication, override, id=f"{shape}-{index}")
            for index, (phrase, medication, (override, shape)) in enumerate(
                itertools.product(
                    BREADTH_PHRASES, BREADTH_MEDICATIONS, BREADTH_OVERRIDES
                )
            )
        ],
    )
    def test_no_document_is_absent_and_complete(
        self, phrase: str, medication: str, override: dict[str, str]
    ) -> None:
        document = (
            "Patient Name: Harold Nkemdirim\n"
            "MRN: RGT/44219/B\n"
            f"{phrase}\n"
            f"{medication}\n"
        )
        try:
            handoff = through_the_boundary(document, QUESTION, **override)
        except HandoffRefused:
            return
        displaced = displaced_typed_facts(handoff)
        report = handoff.facts.completeness()
        assert not (displaced and report.complete), (
            f"{len(displaced)} displaced TYPED_FACT segment(s) at coverage "
            f"{report.coverage():.4f} with the payload stating {report.describe()!r}"
        )
        assert not displaced, (
            "TYPED_FACT segments the projection cannot show: "
            + "; ".join(segment.text.strip() for segment in displaced)
        )


class TestTheAgeOverrideTheSameRootReaches:
    """The MINOR same-root case, measured rather than argued.

    `age` is the one override the design explicitly argues for — a caller
    stating the age is better evidence than a regex — and a band is not the same
    class of loss as a dropped contraindication. It is the same ROOT, so it is
    closed by the same change rather than by an exception for it.
    """

    AGED_LETTER = (
        "Patient Name: Harold Nkemdirim\nAged 84\nComplete heart block\n"
    )

    def test_a_displacing_age_band_is_not_counted_carried(self) -> None:
        try:
            handoff = through_the_boundary(self.AGED_LETTER, QUESTION, age="70")
        except HandoffRefused:
            return
        assert handoff.synthesis_context.age_group == "older_adult_65_74", (
            "the caller's age must still win — the override is approved"
        )
        displaced = displaced_typed_facts(handoff)
        assert not displaced, (
            "the extracted 'Aged 84' span is still counted TYPED_FACT while the "
            "payload renders the caller's band: "
            + "; ".join(segment.text.strip() for segment in displaced)
        )

    def test_an_agreeing_age_is_still_carried(self) -> None:
        """CONTROL. A caller who states the SAME band displaces nothing.

        The property is about the represented meaning, not about whether an
        override happened. A rule that punished the mere presence of a key would
        be pessimism without information, and this control is what distinguishes
        the two: it holds before the remediation and must still hold after it.
        """
        handoff = through_the_boundary(self.AGED_LETTER, QUESTION, age="84")
        assert handoff.synthesis_context.age_group == "older_adult_75_89"
        assert not displaced_typed_facts(handoff)


class TestTheRefusalToResupplyWorkflow:
    """The pathway the product's own refusal message instructs.

    `_WANTED` names `medications (with dose)` and `findings`, and those are the
    two keys that displace the document's extracted clinical content. So the
    product's own remediation advice is the instruction that triggered the
    defect, which is what made it CRITICAL rather than theoretical.
    """

    AMBIGUOUS = (
        "Patient Name: Sarah May Okonkwo\n"
        "NHS Number: 943 476 5919\n"
        "Complete heart block\n"
        "Permanent pacemaker in situ\n"
    )

    def test_the_upload_posture_refuses_the_ambiguous_document(self) -> None:
        """Step 1 of the workflow: the 422 that names the remedy."""
        from mao.core.deident.ambiguity import AmbiguousDocument

        with pytest.raises(AmbiguousDocument):
            with protected_request(RequestProtection(trace_id="test")):
                protect_channel(
                    self.AMBIGUOUS,
                    InputChannel.REPORT,
                    refuse_ambiguity=True,
                    structured={},
                )

    def test_the_resupplied_document_does_not_claim_what_it_dropped(self) -> None:
        """Step 2: re-sent unchanged with the fields the refusal asked for."""
        try:
            handoff = through_the_boundary(
                self.AMBIGUOUS,
                QUESTION,
                patient_name="Sarah May Okonkwo",
                findings="atrial fibrillation",
            )
        except HandoffRefused:
            return
        payload = handoff.synthesis_context.render().lower()
        report = handoff.facts.completeness()
        assert "atrial fibrillation" in payload, "the resupplied field must win"
        if "complete heart block" not in payload:
            assert not report.complete
            assert 3 in report.unresolved_lines, (
                "the heart block on source line 3 is absent from the payload and "
                f"is not named: {report.describe()!r}"
            )


class TestThePropertyOnAPayloadThatActuallyShips:
    """The same properties where the projection PROCESSES rather than refusing.

    Every assertion here is unconditional. There is no `except HandoffRefused`
    escape, because on this letter there is no refusal to escape to — which is
    the point of the letter. If a future change makes these refuse, these tests
    FAIL rather than quietly passing, and that is the correct direction: it
    would mean the remediation had started buying its correctness with
    refusals.
    """

    OVERRIDES = (
        pytest.param(FINDINGS_OVERRIDE, ("Complete heart block",), id="findings"),
        pytest.param(MEDICATIONS_OVERRIDE, ("Warfarin 5mg od",), id="medications"),
        pytest.param(
            BOTH_OVERRIDE,
            ("Complete heart block", "Warfarin 5mg od"),
            id="both",
        ),
    )

    @pytest.mark.parametrize(("structured", "displaced_phrases"), OVERRIDES)
    def test_it_processes_and_reports_itself_incomplete(
        self, structured: dict[str, str], displaced_phrases: tuple[str, ...]
    ) -> None:
        handoff = through_the_boundary(VITALS_LETTER, QUESTION, **structured)
        report = handoff.facts.completeness()
        payload = handoff.synthesis_context.render()
        for phrase in displaced_phrases:
            assert phrase.lower() not in payload.lower(), (
                f"{phrase!r} was expected to be displaced by {structured}"
            )
        assert not report.complete, (
            f"content is absent and the payload states {report.describe()!r}"
        )
        assert "Case completeness: INCOMPLETE" in payload

    @pytest.mark.parametrize(("structured", "displaced_phrases"), OVERRIDES)
    def test_it_names_every_displaced_source_line(
        self, structured: dict[str, str], displaced_phrases: tuple[str, ...]
    ) -> None:
        handoff = through_the_boundary(VITALS_LETTER, QUESTION, **structured)
        report = handoff.facts.completeness()
        lines = VITALS_LETTER.splitlines()
        expected = {
            number
            for number, line in enumerate(lines, start=1)
            for phrase in displaced_phrases
            if phrase in line
        }
        assert expected, "the fixture no longer contains the displaced phrases"
        assert expected <= set(report.unresolved_lines), (
            f"displaced source line(s) {sorted(expected - set(report.unresolved_lines))} "
            f"are not named. Reported: {report.describe()!r}"
        )

    @pytest.mark.parametrize(("structured", "displaced_phrases"), OVERRIDES)
    def test_no_typed_fact_is_absent_from_the_payload(
        self, structured: dict[str, str], displaced_phrases: tuple[str, ...]
    ) -> None:
        handoff = through_the_boundary(VITALS_LETTER, QUESTION, **structured)
        displaced = displaced_typed_facts(handoff)
        assert not displaced, (
            "these segments are counted TYPED_FACT and the projection cannot be "
            "asked to show them: "
            + "; ".join(
                f"line {segment.line + 1} {segment.text.strip()!r}"
                for segment in displaced
            )
        )

    @pytest.mark.parametrize(("structured", "displaced_phrases"), OVERRIDES)
    def test_the_undisplaced_content_is_still_carried(
        self, structured: dict[str, str], displaced_phrases: tuple[str, ...]
    ) -> None:
        """CONTROL. The remediation must cost only what was displaced.

        The vitals are on the same document and no structured key touches them.
        If they stopped being carried, the fix would be pessimism rather than
        accuracy, and every assertion above would be satisfiable by an account
        that had simply given up.
        """
        handoff = through_the_boundary(VITALS_LETTER, QUESTION, **structured)
        assert handoff.synthesis_context.vitals == {
            "blood_pressure": "128/76",
            "heart_rate": "36",
            "temperature": "36.8",
        }
        assert handoff.facts.completeness().carried >= 3

    def test_the_control_letter_is_complete_without_an_override(self) -> None:
        """The isolating variable, on the letter the assertions above use."""
        handoff = through_the_boundary(VITALS_LETTER, QUESTION)
        report = handoff.facts.completeness()
        assert report.complete, report.describe()
        assert not displaced_typed_facts(handoff)
        payload = handoff.synthesis_context.render().lower()
        for phrase in ("complete heart block", "penicillin", "warfarin"):
            assert phrase in payload


class TestTheResolutionAndTheProjectionAreOneModel:
    """The build side and the measure side must not be two opinions.

    `PROJECT_STATE` records the C2 anti-pattern this project has already
    failed: *"a notice stopped being absent, and a second DETECTOR decided when
    it fired, disagreeing with the transformation it described."* ADV19-1 is
    that shape between the account and the projection. A remediation that
    introduced a THIRD description — one for building the context, another for
    measuring it — would be the same defect again.

    So this binds them: what `resolve` decides the projection will carry is what
    `of_case` reads back off the compiled projection, for every field.
    """

    @pytest.mark.parametrize(
        "structured",
        [
            pytest.param({}, id="control-no-override"),
            pytest.param(FINDINGS_OVERRIDE, id="findings-override"),
            pytest.param(MEDICATIONS_OVERRIDE, id="medications-override"),
            pytest.param(BOTH_OVERRIDE, id="both-overrides"),
            pytest.param({"age": "70"}, id="age-override"),
            pytest.param({"age": "not a number"}, id="unparseable-age"),
            pytest.param({"conditions": "AF; CKD"}, id="caller-only-conditions"),
            pytest.param({"findings": " ; ; "}, id="empty-findings-override"),
            pytest.param({"findings": "AF; AF"}, id="repeated-findings-override"),
        ],
    )
    def test_what_is_resolved_is_what_the_compiled_projection_carries(
        self, structured: dict[str, str]
    ) -> None:
        from mao.trust.handoff.compiler import case_from_facts
        from mao.trust.handoff.effective import (
            CALLER_ONLY,
            EXTRACT_BACKED,
            EffectiveProjection,
        )
        from mao.trust.handoff.extract import extract

        facts = extract(DECISIVE_LETTER, question=QUESTION)
        case = case_from_facts(facts, structured=structured, question=QUESTION)
        resolved = EffectiveProjection.resolve(facts, structured)
        compiled = EffectiveProjection.of_case(case, facts)
        for name in EXTRACT_BACKED + CALLER_ONLY:
            assert resolved.values_for(name) == compiled.values_for(name), (
                f"the resolution and the compiled projection disagree about "
                f"{name!r}: {resolved.values_for(name)} vs "
                f"{compiled.values_for(name)}"
            )

    def test_the_compiled_projection_is_what_the_payload_renders(self) -> None:
        """`of_case` reads the context; `SafeSynthesisContext` is built from it.

        Asserted rather than assumed, because the whole closure rests on these
        being the same values. If `compile_synthesis_context` ever built a field
        from something other than the context, the account would go back to
        measuring an object the payload is not.
        """
        from mao.trust.handoff.effective import EffectiveProjection

        handoff = through_the_boundary(DECISIVE_LETTER, QUESTION)
        compiled = EffectiveProjection.of_case(handoff.case, handoff.facts)
        context = handoff.synthesis_context
        assert compiled.values_for("findings") == context.clinically_relevant_findings
        assert compiled.values_for("medications") == context.medications_and_doses
        assert compiled.values_for("conditions") == context.conditions
        assert compiled.values_for("allergies") == context.allergies
        assert compiled.values_for("uncertainties") == context.uncertainties
        assert compiled.values_for("age_group") == (
            (context.age_group,) if context.age_group else ()
        )

    def test_an_unparseable_age_establishes_nothing_and_displaces_nothing(
        self,
    ) -> None:
        """CONTROL. A structured value that is not a value is not an override."""
        aged = "Patient Name: Harold Nkemdirim\nAged 84\nComplete heart block\n"
        handoff = through_the_boundary(aged, QUESTION, age="unknown")
        assert handoff.synthesis_context.age_group == "older_adult_75_89"
        assert not displaced_typed_facts(handoff)


class TestTheOverrideSemanticsAtTheSeam:
    """What counts as "still represented", probed at the edges.

    The property is about REPRESENTED MEANING, not about whether a structured
    key was present. These pin the boundary so that a later change cannot make
    the account more generous without a test saying so.
    """

    LETTER = (
        "Patient Name: Harold Nkemdirim\n"
        "Complete heart block\n"
        "Warfarin 5mg od\n"
        "BP 128/76\n"
        "HR 36\n"
    )

    def _reading(self, **structured: str):
        handoff = through_the_boundary(self.LETTER, QUESTION, **structured)
        return handoff, handoff.facts.completeness()

    def test_an_exact_restatement_displaces_nothing(self) -> None:
        handoff, report = self._reading(findings="Complete heart block")
        assert report.complete, report.describe()
        assert not displaced_typed_facts(handoff)

    def test_a_superset_keeps_the_extracted_value_represented(self) -> None:
        handoff, report = self._reading(
            findings="Complete heart block; Mild cognitive impairment"
        )
        assert report.complete, report.describe()
        assert not displaced_typed_facts(handoff)
        assert handoff.synthesis_context.clinically_relevant_findings == (
            "Complete heart block",
            "Mild cognitive impairment",
        )

    @pytest.mark.parametrize(
        "stated",
        [
            pytest.param("complete heart block", id="case-differs"),
            pytest.param("Complete heart block (resolved)", id="qualified"),
        ],
    )
    def test_a_near_miss_is_accounted_rather_than_assumed_equivalent(
        self, stated: str
    ) -> None:
        """R-4, and it is deliberate.

        A qualified finding is a DIFFERENT clinical claim, and the payload
        genuinely no longer shows what the document said. Treating it as
        equivalent would need a similarity judgement about clinical text, which
        is the move every previous wave got wrong. The safe direction is to
        account it and name the line.
        """
        handoff, report = self._reading(findings=stated)
        assert not report.complete
        assert 2 in report.unresolved_lines, report.describe()
        assert not displaced_typed_facts(handoff)

    def test_a_repeated_structured_value_is_not_shown_twice(self) -> None:
        handoff, _ = self._reading(findings="AF; AF; AF")
        assert handoff.synthesis_context.clinically_relevant_findings == ("AF",)

    def test_an_empty_override_establishes_nothing(self) -> None:
        """Whitespace is not a value, so it is not an override."""
        handoff, report = self._reading(findings="   ;  ; ")
        assert report.complete, report.describe()
        assert handoff.synthesis_context.clinically_relevant_findings == (
            "Complete heart block",
        )

    @pytest.mark.parametrize(
        "structured",
        [
            pytest.param({"conditions": "CKD stage 3"}, id="conditions"),
            pytest.param({"allergies": "penicillin"}, id="allergies"),
            pytest.param({"uncertainties": "adherence unclear"}, id="uncertainties"),
        ],
    )
    def test_a_caller_only_key_displaces_nothing(
        self, structured: dict[str, str]
    ) -> None:
        handoff, report = self._reading(**structured)
        assert report.complete, report.describe()
        assert not displaced_typed_facts(handoff)

    def test_a_key_for_a_non_overridable_field_takes_no_effect(self) -> None:
        """`vitals` is built from the extractor only.

        A caller key that silently took effect here would be a new displacement
        route, and one that silently did nothing while appearing to work would
        be its own defect. It does nothing, and the extracted vitals stand.
        """
        handoff, report = self._reading(vitals="HR 200")
        assert handoff.synthesis_context.vitals == {
            "blood_pressure": "128/76",
            "heart_rate": "36",
        }
        assert report.complete, report.describe()


def _violations_over_the_population() -> int:
    """The core property, measured over the decisive population.

    Returns the number of documents in which a `TYPED_FACT` segment is absent
    from the payload that ships. Zero as shipped; a mutation control that does
    not move this number off zero is not a control.
    """
    documents = 0
    violations = 0
    for override in (FINDINGS_OVERRIDE, MEDICATIONS_OVERRIDE, BOTH_OVERRIDE):
        for phrase in BREADTH_PHRASES:
            document = (
                "Patient Name: Harold Nkemdirim\n"
                "MRN: RGT/44219/B\n"
                f"{phrase}\n"
                "Warfarin 5mg od\n"
            )
            try:
                handoff = through_the_boundary(document, QUESTION, **override)
            except HandoffRefused:
                continue
            documents += 1
            if displaced_typed_facts(handoff):
                violations += 1
    assert documents, (
        "every document in the population refused, so this measurement is "
        "vacuous and cannot distinguish a fix from a mutant"
    )
    return violations


class TestMutationControls:
    """A clean result is worth nothing unless it can be made dirty.

    `00_RULES` and four consecutive gates say the same thing: a green oracle
    that cannot fail is not evidence. These restore the `ea46e7e` behaviour IN
    MEMORY — no file is modified — and require the property to break.
    """

    def test_the_population_is_not_vacuous(self) -> None:
        """The baseline, and the reason the mutants below mean anything."""
        assert _violations_over_the_population() == 0

    def test_reverting_to_an_account_unaware_override_breaks_the_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTANT 1 — exactly `ea46e7e`.

        The compiler still resolves the structured field and still ships the
        caller's value; it simply does not re-state the account against the
        projection. This is the defect verbatim, and the oracle must fail on it.
        """
        from mao.trust.handoff import compiler

        monkeypatch.setattr(
            compiler, "accounted_projection", lambda context, facts: facts
        )
        assert _violations_over_the_population() > 0, (
            "the account-unaware override was restored and the oracle still "
            "reported zero violations — it is not testing the closure"
        )

    def test_a_vacuous_effective_model_breaks_the_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTANT 2 — the model exists but answers yes to everything.

        A projection that claims to represent whatever it is asked about is the
        same failure one layer in: the seam is called, the account is re-stated,
        and nothing changes. Catching this is what distinguishes testing the
        CLOSURE from testing that a function is called.
        """
        from mao.trust.handoff.effective import EffectiveProjection

        monkeypatch.setattr(
            EffectiveProjection,
            "represents",
            lambda self, field, value: True,
        )
        assert _violations_over_the_population() > 0, (
            "a projection that represents everything still reported zero "
            "violations — the oracle is asking the mechanism to grade itself"
        )

    def test_a_naive_merge_is_not_what_closes_this(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTANT 3 — the fix the reviews explicitly ruled out.

        Merging the caller's values with the extractor's makes the displaced
        content present again, so the absent-and-complete property goes quiet —
        which is precisely why a merge must not be mistaken for closure. What it
        does NOT do is keep the projection honest: it manufactures a payload
        asserting both a clinician's correction and the text it corrects.
        """
        from mao.trust.handoff import effective

        original = effective._field

        def merging(name, stated, extracted):
            field = original(name, stated, extracted)
            if not field.displaced:
                return field
            return effective.EffectiveField(
                name=field.name,
                values=field.values + field.displaced,
                source=field.source,
            )

        monkeypatch.setattr(effective, "_field", merging)
        handoff = through_the_boundary(
            DECISIVE_LETTER, QUESTION, **FINDINGS_OVERRIDE
        )
        findings = handoff.synthesis_context.clinically_relevant_findings
        assert "mild cognitive impairment" in findings
        assert "Complete heart block" in findings, (
            "the merge mutant did not take effect, so this control proves nothing"
        )
        # The payload now presents the caller's correction AND the text it
        # corrects as two coexisting facts about one patient. That is the
        # outcome `TestNoNaiveMergeOfCallerAndExtractedValues` forbids, and this
        # control is what shows those assertions can fail.


class TestAdjacentClosuresAreUndisturbed:
    """Regression obligations at the boundary this remediation changed.

    Every closure both `46a198a` reviewers and both `ea46e7e` reviewers
    confirmed is a regression obligation under `00_RULES` re-gate scope item 3.
    These are the ones adjacent to the account and the compiler, re-asserted
    here so a change to the accounting cannot quietly move them.
    """

    def test_presentation_syntax_still_decides_nothing(self) -> None:
        """AR18-1 / ADV18-1, closed by removal at `ea46e7e`.

        The one-character isolating control: with and without the `#`, the
        accounts must be identical. Re-stating the account against the
        projection must not reintroduce a reading of how a line is spelled.
        """
        # A carried line is present so that BOTH variants process rather than
        # refusing on coverage: two documents that both refuse would compare
        # equal without the property having been exercised.
        with_hash = through_the_boundary(
            "Patient Name: Harold Nkemdirim\n"
            "Warfarin 5mg od\n"
            "# Permanent pacemaker in situ\n",
            QUESTION,
        )
        without = through_the_boundary(
            "Patient Name: Harold Nkemdirim\n"
            "Warfarin 5mg od\n"
            "Permanent pacemaker in situ\n",
            QUESTION,
        )
        left, right = with_hash.facts.completeness(), without.facts.completeness()
        assert (left.carried, left.unresolved, left.unresolved_lines) == (
            right.carried,
            right.unresolved,
            right.unresolved_lines,
        ), "the '#' changed the account again"
        assert left.unresolved, (
            "neither document has unresolved residue, so the isolating control "
            "compares two empty accounts and proves nothing"
        )

    def test_the_colon_route_is_still_carried(self) -> None:
        """AR17-1: clinical content on the LABEL side of a colon."""
        handoff = through_the_boundary(
            "Patient Name: Harold Nkemdirim\n"
            "Complete heart block, permanent pacemaker implanted: 4 March 2019\n",
            QUESTION,
        )
        assert "Complete heart block" in handoff.synthesis_context.render()

    def test_the_residue_still_crosses_no_boundary(self) -> None:
        """AR17-2 / invariant 12's CONTENT half.

        Measuring the loss and transmitting the residue are different
        requirements. Making the account see MORE loss must not make the payload
        carry any of it.
        """
        letter = (
            "Patient Name: Harold Nkemdirim\n"
            "Dictated by Sarah Thompson, medical secretary, ext 4471.\n"
            "Complete heart block\n"
            "Warfarin 5mg od\n"
        )
        try:
            handoff = through_the_boundary(letter, QUESTION, **BOTH_OVERRIDE)
        except HandoffRefused:
            handoff = through_the_boundary(letter, QUESTION)
        payload = handoff.synthesis_context.render()
        for token in ("Sarah Thompson", "4471", "secretary"):
            assert token not in payload, f"{token!r} reached the external payload"
        assert handoff.facts.unresolved_text(), (
            "the residue is no longer measurable locally, so this assertion "
            "could pass by there being nothing to leak"
        )

    def test_the_account_still_reads_no_source_text_predicate(self) -> None:
        """The structural guard ADV18-1's closure rests on, re-checked.

        `against()` re-states segments from a projection's answer, never from
        the text of the segment. If a pattern over source text appeared in the
        module, the class that defeated four gates would be open again.
        """
        import re as _re

        from mao.trust.handoff import accounting as accounting_module

        patterns = [
            name
            for name, value in vars(accounting_module).items()
            if isinstance(value, _re.Pattern)
        ]
        assert not patterns, f"a source-text pattern reappeared: {patterns}"
