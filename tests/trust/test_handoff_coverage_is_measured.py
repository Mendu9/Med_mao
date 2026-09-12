r"""A-1 and ADV16-6: the Safe Handoff's loss guard was blind to its own failure.

The two reviews found this independently, from opposite directions, and BOTH
mechanisms had to be fixed because either one alone leaves the guard useless.

    A-1      `extract.py` read
                 if _DOSE.search(stripped):        ...; matched = True
                 elif _is_clinical_line(stripped): ...; matched = True
                 if not matched and _is_clinical_line(stripped): uncovered.append(...)
             If `_is_clinical_line` is true the `elif` already ran and `matched`
             is true, so the final guard is unsatisfiable and `uncovered` was
             unconditionally empty. Dead code.

    ADV16-6  the denominator consulted the SAME lexicon as the de-identifier,
             so a clinical line that lexicon does not recognise was in NEITHER
             the numerator NOR the denominator - not "uncovered" but invisible.
             `_is_clinical_line` is True for 0 of the predecessor's 49 clinical
             phrases.

The load-bearing case below is the adversarial reviewer's: an ordinary clinic
letter with no patient name, so the de-identification refusal does not fire,
asked whether donepezil - a bradycardic drug - is safe. The single most
decision-relevant fact in the document is an implanted pacemaker, and it was
dropped from the projection while `coverage()` reported 1.00, `uncertainties`
reported `()`, and `clinical_coverage: 1.0` was returned to the caller, cached
in Redis and written to a trace.

The fix is a MEASUREMENT, not a vocabulary. Widening the lexicon so it
recognises 'Medtronic Azure Pacemaker' would make this file pass and would be
the seventh round of exactly what `00_RULES.md` forbids.
"""
from __future__ import annotations

import dataclasses

import pytest

from mao.trust.handoff.compiler import (
    MINIMUM_COVERAGE,
    HandoffRefused,
    compile_handoff,
)
from mao.trust.handoff.extract import extract

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


class TestCoverageMeasuresTheDocumentNotTheLexicon:
    """ADV16-6."""

    def test_the_pacemaker_line_is_counted_somewhere(self) -> None:
        facts = extract(LETTER)
        assert PACEMAKER in (facts.findings + facts.uncovered), (
            "the line is in neither the numerator nor the denominator, so no "
            "threshold on coverage() could ever see it"
        )

    def test_coverage_is_below_one_when_content_was_not_carried(self) -> None:
        facts = extract(LETTER)
        assert facts.coverage() < 1.0, (
            f"coverage() reported {facts.coverage()} on a document that lost "
            "content. That is the ADV16-6 mechanism: 1.00 precisely when the "
            "loss is total."
        )

    def test_the_lost_lines_are_named_not_merely_counted(self) -> None:
        facts = extract(LETTER)
        assert facts.uncovered, "nothing is reported as not carried"

    def test_uncovered_is_reachable_at_all(self) -> None:
        """A-1: the branch that populates it was unsatisfiable."""
        facts = extract("qwertyuiop asdfghjkl\nzxcvbnm poiuytrewq\n")
        assert facts.uncovered, "uncovered is still unconditionally empty"

    def test_the_denominator_does_not_consult_the_lexicon(self) -> None:
        """The property, stated directly. Two documents with the same number of
        content lines and the same number of typed facts must score the same,
        whatever vocabulary they happen to use."""
        known = extract("Donepezil 10 mg once daily.\nWibble wobble flim flam.\n")
        unknown = extract("Donepezil 10 mg once daily.\nQwerty uiop asdf ghjk.\n")
        assert known.coverage() == unknown.coverage()


class TestARedactedIdentifierLineIsNotCountedAsLoss:
    """The trap the architecture review predicted in its A-1 remediation note:
    "with the catch-all removed, every clinical line currently carried verbatim
    into findings becomes uncovered, coverage collapses, and the 50% threshold
    starts refusing ordinary documents."

    It is real and it fired. An ordinary memory-clinic letterhead is five
    identifier lines and two content lines; counting `Patient Name: [NAME]` as
    content the projection failed to carry scored it 0.286 and refused a
    perfectly answerable document. That measures the DE-IDENTIFIER's success as
    the EXTRACTOR's failure.

    The exclusion is structural - a placeholder this system emitted accounts for
    the whole of the line's value - and consults no vocabulary, which is the
    only reason it is admissible.
    """

    REPORT = (
        "MEMORY CLINIC REPORT\n"
        "Patient Name: [NAME]\n"
        "MRN: [MRN]\n"
        "Address: [ADDRESS]\n"
        "DOB: [DOB]\n"
        "Contact: [EMAIL]\n"
        "Findings: moderate hippocampal atrophy, MMSE 21/30.\n"
    )

    def test_an_ordinary_letterhead_scores_full_coverage(self) -> None:
        facts = extract(self.REPORT)
        assert facts.coverage() == 1.0, (
            f"an ordinary de-identified letterhead scored {facts.coverage()}; "
            "the identifier lines are being counted as clinical content lost"
        )

    def test_and_is_therefore_not_refused(self) -> None:
        handoff = compile_handoff(self.REPORT, question="Summarise this report")
        assert handoff.synthesis_context is not None

    def test_the_clinical_line_is_still_carried(self) -> None:
        """Non-vacuity: coverage is 1.0 because the content WAS carried, not
        because the document was scored as having none."""
        facts = extract(self.REPORT)
        assert facts.content_lines == 2
        assert any("hippocampal atrophy" in f for f in facts.findings)

    def test_a_line_with_a_placeholder_AND_content_still_counts(self) -> None:
        """The exclusion must be for lines whose WHOLE value was an identifier,
        never for a clinical line that happens to contain a placeholder."""
        facts = extract("Seen with [NAME] for Medtronic Azure Pacemaker check.\n")
        assert facts.content_lines == 1


class TestCoverageIsOneOnlyWhenThereWasNothingToLose:
    def test_an_empty_document_scores_one(self) -> None:
        assert extract("").coverage() == 1.0

    def test_a_whitespace_document_scores_one(self) -> None:
        assert extract("\n\n   \n").coverage() == 1.0

    def test_a_document_whose_lines_all_became_facts_scores_one(self) -> None:
        facts = extract("Donepezil 10 mg once daily.\nBlood pressure 128/76.\n")
        assert facts.coverage() == 1.0
        assert facts.uncovered == ()


class TestTheMeasurementIsNonVacuous:
    """If `uncovered` is forced empty the assertions above must stop holding.

    00_RULES prefers a mutation test for any claim that a guard would detect a
    bypass. This is that mutation: it re-creates the b63311d behaviour exactly
    - `uncovered` unconditionally `()` - and shows the suite notices.
    """

    def test_forcing_uncovered_empty_restores_the_defect(self) -> None:
        facts = extract(LETTER)
        assert facts.coverage() < 1.0
        as_it_was = dataclasses.replace(facts, uncovered=())
        assert as_it_was.coverage() == 1.0, (
            "with uncovered empty the function must report 1.00 again - if it "
            "does not, this test is not measuring what it claims to"
        )


class TestTheThinProjectionRefusalCanFire:
    """A-1 consequence 1: `_require_substance` compared a constant 1.0 to 0.5,
    so the entire HandoffRefused 'thin projection' branch was unreachable and
    only the `has_clinical_facts()` 'empty' branch was live."""

    def test_a_document_the_extractor_mostly_cannot_read_is_refused(self) -> None:
        unreadable = "Donepezil 10 mg once daily.\n" + "\n".join(
            f"qwertyuiop{n} asdfghjkl{n} zxcvbnm{n}" for n in range(10)
        )
        with pytest.raises(HandoffRefused):
            compile_handoff(unreadable, question="Is donepezil safe?")

    def test_the_refusal_names_the_fields_that_would_resolve_it(self) -> None:
        unreadable = "Donepezil 10 mg once daily.\n" + "\n".join(
            f"qwertyuiop{n} asdfghjkl{n} zxcvbnm{n}" for n in range(10)
        )
        with pytest.raises(HandoffRefused) as raised:
            compile_handoff(unreadable, question="Is donepezil safe?")
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
        """Pinned so a later reader does not mistake it for a measurement. It
        was unreachable at b63311d, so no operating experience stands behind
        it, and it is deliberately NOT tuned to make any particular document
        refuse - doing that would make the threshold a function of whichever
        examples someone happened to try."""
        assert MINIMUM_COVERAGE == 0.5


class TestTheModelIsToldWhatCouldNotBeCarried:
    """A-1 consequence 2. `providers/gateway.py` tells the model in its system
    framing that "facts that could not be carried safely are absent rather than
    summarised... Where a fact you would need is missing, say which fact and do
    not assume it." At b63311d `uncertainties` was structurally always empty,
    so the model was told to look for a signal that was never sent - and
    `uncertainties=()` positively asserts there were none."""

    def test_uncertainties_is_populated(self) -> None:
        handoff = compile_handoff(LETTER, question="Is donepezil safe?")
        assert handoff.synthesis_context.uncertainties, (
            "the projection still asserts that nothing was left behind"
        )

    def test_the_uncertainties_reach_the_rendered_payload(self) -> None:
        handoff = compile_handoff(LETTER, question="Is donepezil safe?")
        rendered = handoff.synthesis_context.render()
        assert any(
            item in rendered for item in handoff.synthesis_context.uncertainties
        ), "the model is not told what the projection could not carry"
