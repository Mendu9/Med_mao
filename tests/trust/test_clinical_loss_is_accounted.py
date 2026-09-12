r"""O3 — the AR17/ADV17 documents cannot report false completeness.

## The property, stated as the disjunction that matters

For every clinically material phrase in the source, EITHER

    (a) it is represented in `synthesis_context.render()` — the text the
        external synthesis model actually receives; OR

    (b) the projection reports `complete is False` AND names the SOURCE LINE
        the phrase came from in `completeness.unresolved_lines`.

Never both absent and complete. That conjunction — content gone, projection
claiming to be whole — is AR17-1 and ADV17-2, and it is the single failure this
remediation exists to make unreachable. At `ff34722` the measured outcome was
`coverage 1.0000`, `uncertainties` empty, no notice, and Complete Heart Block
plus an implanted pacemaker missing from the projection handed to a model asked
whether a bradycardic drug was safe.

Note what the property deliberately does NOT require: that everything is
carried. Phase 1 cannot decide whether an unresolved segment mattered. It
requires only that the system never claims to have carried what it did not.

## How this oracle is independent

  the documents are fixed, and are the ones the reviews measured
      Not generated, not chosen by me. They are quoted verbatim from the AR17
      and ADV17 findings, so the corpus cannot have been tuned.

  the material phrases are declared by hand from the SOURCE
      A clinician reading these letters names the pacemaker, the heart block,
      the anaphylaxis, the anticoagulant. That list is written below as a
      constant. It is not computed by `extract`, not derived from `findings`,
      and not read out of the lexicon — all three of which are the machinery
      under test, and the last of which is the ADV16-6 mechanism by which
      "is this line clinical?" kept being answered by the thing that was wrong.

  the expected line number is computed by counting newlines in the raw text
      `_source_line_of` below splits the raw document and finds the phrase. It
      never asks `accounting`, `split_lines` or the `CompletenessReport` where
      a line is. So "the report names the right line" is checked against an
      independently derived answer.

  the boundary is the real one
      `protect_channel(... InputChannel.REPORT ...)` inside `protected_request`,
      then `compile_handoff(..., events=tuple(protected.events))` — the same
      three calls `clinical_agent._handle_pdf_report` makes.
"""
from __future__ import annotations

import pytest

from mao.trust.classes import InputChannel
from mao.trust.egress.gateway import RequestProtection, protected_request
from mao.trust.handoff.accounting import (
    CompletenessReport,
    Segment,
    SegmentState,
    SourceAccounting,
)
from mao.trust.handoff.compiler import compile_handoff
from mao.trust.handoff.extract import ExtractedFacts, extract
from mao.trust.inputs.boundary import protect_channel

# --------------------------------------------------------------------------
# The two documents, verbatim.
# --------------------------------------------------------------------------

AR17_DOCUMENT = (
    "Patient Name: Harold James Nkemdirim\n"
    "MRN: RGT/44219/B\n"
    "Medtronic Azure Pacemaker in situ, lead checked by: Dr Aoife Rankin\n"
    "Complete Heart Block confirmed, discussed with next of kin: Sarah Okonkwo\n"
    "Penicillin anaphylaxis, alert added by: Dr John Smith\n"
    "Warfarin INR 4.8, phoned through to: 0113 496 0231\n"
    "Is donepezil safe for this patient?\n"
)

ADV17_DOCUMENT = (
    "MEMORY CLINIC REVIEW\n"
    "Patient Name: Harold Nkemdirim\n"
    "MRN: A1234567\n"
    "Complete heart block, permanent pacemaker implanted: 14/03/2019\n"
    "Warfarin stopped after recurrent falls on: 02/06/2023\n"
    "MMSE 21/30 at review.\n"
    "Donepezil 10mg od commenced.\n"
    "Aged 84.\n"
)

QUESTION = "Is donepezil safe for this patient?"

#: Declared by hand from the source. Donepezil is a cholinesterase inhibitor and
#: therefore bradycardic; complete heart block and an implanted pacemaker are
#: contraindications, penicillin anaphylaxis governs any co-prescription, and an
#: INR of 4.8 on warfarin is itself actionable. A model asked whether donepezil
#: is safe and told none of this would answer confidently and wrongly.
MATERIAL_PHRASES = {
    "AR17": (
        "Medtronic Azure Pacemaker in situ",
        "Complete Heart Block confirmed",
        "Penicillin anaphylaxis",
        "Warfarin INR 4.8",
    ),
    "ADV17": (
        "Complete heart block, permanent pacemaker implanted",
        "Warfarin stopped after recurrent falls",
        "MMSE 21/30 at review.",
        "Donepezil 10mg od commenced.",
    ),
}

DOCUMENTS = {"AR17": AR17_DOCUMENT, "ADV17": ADV17_DOCUMENT}


# --------------------------------------------------------------------------
# Independent helpers
# --------------------------------------------------------------------------


def _source_line_of(document: str, phrase: str) -> int:
    """The 1-based line of `document` on which `phrase` starts.

    Counted here, by splitting on newlines, so the assertion about
    `unresolved_lines` is checked against a number this test derived and not
    against one the implementation offered.
    """
    for index, line in enumerate(document.split("\n"), start=1):
        if phrase in line:
            return index
    raise AssertionError(f"{phrase!r} is not in the document — fix the corpus")


def _take_through_the_real_boundary(document: str, label: str):
    with protected_request(RequestProtection(trace_id=f"o3-{label}")):
        protected = protect_channel(
            document,
            InputChannel.REPORT,
            refuse_ambiguity=False,
            structured=None,
        )
    return compile_handoff(
        protected.text,
        question=QUESTION,
        events=tuple(protected.events),
    )


def _is_represented(phrase: str, rendered: str) -> bool:
    """Whether the model can read this phrase out of the payload.

    Compared with the identifiers taken out, because the boundary legitimately
    replaced them: `discussed with next of kin: Sarah Okonkwo` reaches the model
    as `discussed with next of kin: [NAME]`, and demanding the name back would
    be demanding the leak. Everything up to the removed value must be there.
    """
    lowered = rendered.lower()
    if phrase.lower() in lowered:
        return True
    # Allow the phrase to be terminated by a placeholder the boundary wrote.
    head = phrase.split(":")[0].strip().lower()
    return bool(head) and head in lowered


@pytest.fixture(scope="module")
def handoffs():
    return {label: _take_through_the_real_boundary(doc, label)
            for label, doc in DOCUMENTS.items()}


# --------------------------------------------------------------------------
# The invariant
# --------------------------------------------------------------------------


class TestNothingIsBothAbsentAndReportedComplete:
    """The disjunction, phrase by phrase, on both documents."""

    @pytest.mark.parametrize(
        ("label", "phrase"),
        [(label, phrase)
         for label, phrases in MATERIAL_PHRASES.items()
         for phrase in phrases],
    )
    def test_each_material_phrase_is_carried_or_named(
        self, handoffs, label: str, phrase: str
    ) -> None:
        handoff = handoffs[label]
        rendered = handoff.synthesis_context.render()
        completeness = handoff.synthesis_context.completeness

        if _is_represented(phrase, rendered):
            return

        # Absent. Then the projection must say so, and say WHERE.
        assert completeness.complete is False, (
            f"{label}: {phrase!r} is absent from the payload the model "
            f"receives, and the projection reports itself COMPLETE. This is "
            f"the AR17-1 / ADV17-2 failure exactly.\n  render:\n{rendered}"
        )
        expected_line = _source_line_of(DOCUMENTS[label], phrase)
        assert expected_line in completeness.unresolved_lines, (
            f"{label}: {phrase!r} is absent and the projection is incomplete, "
            f"but source line {expected_line} is not among the lines it names "
            f"({completeness.unresolved_lines}). A clinician told 'something is "
            f"missing' but pointed at the wrong line cannot act on it."
        )

    @pytest.mark.parametrize("label", sorted(DOCUMENTS))
    def test_render_always_contains_a_completeness_statement(
        self, handoffs, label: str
    ) -> None:
        """Unconditional and second, per the class contract."""
        rendered = handoffs[label].synthesis_context.render()
        lines = rendered.split("\n")
        assert len(lines) >= 2, f"{label}: render() is too short to carry one"
        assert lines[1].startswith("Case completeness:"), (
            f"{label}: the completeness statement is not the second line: "
            f"{lines[:2]!r}"
        )
        assert lines[1].strip() != "Case completeness:", (
            f"{label}: the completeness statement is empty"
        )

    @pytest.mark.parametrize("label", sorted(DOCUMENTS))
    def test_an_incomplete_projection_says_so_in_words_a_model_would_act_on(
        self, handoffs, label: str
    ) -> None:
        context = handoffs[label].synthesis_context
        rendered = context.render()
        if context.completeness.complete:
            pytest.skip(f"{label} projects completely; nothing to assert here")
        statement = rendered.split("\n")[1]
        assert "INCOMPLETE" in statement, (
            f"{label}: an incomplete projection does not use the word: "
            f"{statement!r}"
        )
        for instruction in ("NOT included", "do not assume"):
            assert instruction.lower() in statement.lower(), (
                f"{label}: the statement does not tell the model what to do "
                f"about the omission ({instruction!r} missing): {statement!r}"
            )
        assert any(
            str(line) in statement for line in context.completeness.unresolved_lines
        ), f"{label}: the statement names no source line: {statement!r}"

    def test_ar17_is_the_incomplete_one_and_adv17_the_complete_one(
        self, handoffs
    ) -> None:
        """BOTH populations occur.

        A corpus in which every document reports the same status grades nothing:
        an implementation hardcoding `complete = False` would pass the
        disjunction on every case. One of these two must be complete and the
        other must not.
        """
        statuses = {
            label: handoffs[label].synthesis_context.completeness.complete
            for label in DOCUMENTS
        }
        assert set(statuses.values()) == {True, False}, (
            f"both documents report the same completeness status ({statuses}), "
            "so this file cannot tell a real account from a constant"
        )

    def test_the_unresolved_report_carries_no_source_text(self, handoffs) -> None:
        """The AR17-2 half: measuring the loss must not transmit the residue."""
        for label, handoff in handoffs.items():
            rendered = handoff.synthesis_context.render()
            for residue in handoff.facts.unresolved_text():
                stripped = residue.strip().rstrip(":").strip()
                if len(stripped) < 12:
                    continue
                assert stripped not in rendered, (
                    f"{label}: unresolved source text was transmitted verbatim "
                    f"in the payload: {stripped!r}"
                )


class TestTheClinicianTypedBracketToken:
    r"""The second route into the same exclusion, closed.

    `\[[A-Z_]+\]` was the regex by which the old accounting decided a span was
    "an identifier this boundary removed". A clinician's `Pacing mode: [DDDR]`
    satisfies it, so a pacing mode was accounted as a redaction and the whole
    line vanished from the measurement — neither carried nor counted as lost.

    The requirement is that the line is ACCOUNTABLE: it must land in `carried`
    or in `unresolved`, and must not be written off as `IDENTIFIER_REMOVED`.
    """

    LINE = "Pacing mode: [DDDR]"

    def test_the_line_is_accountable(self) -> None:
        facts = extract(self.LINE)
        report = facts.completeness()
        assert report.accountable >= 1, (
            "the clinician's bracketed token was accounted as something other "
            f"than clinical content: {report!r}"
        )

    def test_it_is_not_recorded_as_a_redaction_this_system_performed(self) -> None:
        facts = extract(self.LINE)
        removed = facts.accounting.in_state(SegmentState.IDENTIFIER_REMOVED)
        assert removed == (), (
            "a clinician-typed bracketed token was mistaken for a redaction "
            f"this boundary performed: {[s.text for s in removed]!r}"
        )
        assert report_mentions(facts, self.LINE), (
            "the line is in neither the carried nor the unresolved population"
        )

    def test_the_same_holds_with_no_event_record_at_all(self) -> None:
        """`extract` is given no events here, which is the pessimistic path.

        A missing record must never make a projection look MORE complete. With
        nothing recorded, the line is unresolved — which is the safe direction.
        """
        facts = extract(self.LINE, events=())
        assert facts.completeness().complete is False
        assert facts.completeness().unresolved_lines == (1,)


def report_mentions(facts: ExtractedFacts, line: str) -> bool:
    """Whether `line` is accounted as carried or as unresolved."""
    states = {
        segment.state
        for segment in facts.accounting.segments
        if segment.text.strip() and segment.text.strip() in line
    }
    return bool(states & {SegmentState.TYPED_FACT, SegmentState.UNRESOLVED})


class TestALineCarriedOnlyAsANumberIsStillAccounted:
    r"""W14-2, CLOSED — a partial extraction was accounted as a whole.

    `accounting.account()` decides `TYPED_FACT` from one boolean: whether
    `carried[line_index]` is set. `extract` sets it as soon as the line yields
    ANY field — including when the only thing it yielded is a two-digit number
    pulled out by a `_VITALS` or `_LABS` pattern.

    So a line the projection carries as `heart_rate 36` and nothing else is
    accounted as fully represented, and `CompletenessReport.complete` is True.
    The words around the number never reach `render()`, and the payload tells
    the model:

        "Complete: all N clinically accountable source segment(s) are
         represented in this projection."

    Measured on the two lines below, both taken through the real boundary.

    ## Why this is the same defect class, not a new one

    `extract` has two routes into `findings`: `_DOSE` (a shape) and
    `_is_clinical_line` (a LEXICON lookup). When the lexicon recognises the
    line it is carried whole and nothing is lost. When it does not, only the
    number is carried and the rest is silently dropped — but the account is told
    the line was carried either way.

    That makes whether a loss is DETECTED a function of the clinical vocabulary,
    which is ADV16-6 exactly: `accounting.py`'s own docstring says the whole
    point of the account is that "no vocabulary decides it". A vocabulary still
    decides it, one layer up, in the `carried` map the account is handed.

    ## What a fix would look like — NOT applied here

    `carried` would have to record what was carried, not merely that something
    was, so a line yielding only a number is accounted as partly unresolved.
    That is a change to `mao/trust/handoff/`, which this worktree may not make.

    The two `xfail(strict=True)` cases below are the full invariant, unmodified.
    """

    #: Realistic lines whose vocabulary the clinical lexicon does not contain,
    #: each carrying a vital the extractor does recognise.
    LOSSY_LINES = (
        "Contraindicated per cardiology, HR 36",
        "Cardiology advise against starting, HR 34",
    )

    @staticmethod
    def _handoff(extra: str):
        document = (
            "Patient Name: Harold Nkemdirim\n"
            "MRN: A1234567\n"
            "Donepezil 10mg od commenced.\n"
            f"{extra}\n"
        )
        with protected_request(RequestProtection(trace_id="o3-lossy")):
            protected = protect_channel(
                document, InputChannel.REPORT, refuse_ambiguity=False
            )
        return compile_handoff(
            protected.text, question=QUESTION, events=tuple(protected.events)
        ), document

    @pytest.mark.parametrize("extra", LOSSY_LINES)
    def test_the_words_around_the_number_are_carried_or_named(
        self, extra: str
    ) -> None:
        handoff, document = self._handoff(extra)
        rendered = handoff.synthesis_context.render()
        completeness = handoff.synthesis_context.completeness
        phrase = extra.rsplit(",", 1)[0].strip()

        if phrase.lower() in rendered.lower():
            return
        assert completeness.complete is False, (
            f"{phrase!r} is absent from the payload and the projection reports "
            f"itself COMPLETE.\n  render:\n{rendered}"
        )
        assert _source_line_of(document, extra) in completeness.unresolved_lines

    def test_the_lexicon_is_what_decides_whether_the_loss_is_seen(self) -> None:
        """The ADV16-6 shape, stated directly.

        Same sentence structure, same vital, same document. The only difference
        is whether the clinical lexicon happens to contain a word in the line —
        and that is what decides whether the content survives.
        """
        from mao.trust.handoff.extract import _is_clinical_line

        lossy = "Contraindicated per cardiology, HR 36"
        safe = "Stop request from the cardiologist, HR 38"
        assert _is_clinical_line(lossy) is False
        assert _is_clinical_line(safe) is True

        lossy_render = self._handoff(lossy)[0].synthesis_context.render()
        safe_render = self._handoff(safe)[0].synthesis_context.render()
        assert "Contraindicated per cardiology" not in lossy_render
        assert "Stop request from the cardiologist" in safe_render


class TestTheAssertionCanFail:
    """NON-VACUITY CONTROL for `complete`.

    Every assertion above leans on `CompletenessReport.complete` being a real
    function of the account. These two construct the account directly — one with
    an unresolved segment, one without — and show the flag follows. If it did
    not, the disjunction above would be satisfied by a constant.
    """

    @staticmethod
    def _facts(states: list[SegmentState]) -> ExtractedFacts:
        return ExtractedFacts(
            accounting=SourceAccounting(
                segments=tuple(
                    Segment(
                        line=index,
                        start=index * 10,
                        end=index * 10 + 9,
                        state=state,
                        text=f"segment {index}",
                        carried_as="findings" if state is SegmentState.TYPED_FACT else "",
                    )
                    for index, state in enumerate(states)
                )
            )
        )

    def test_an_unresolved_segment_makes_the_projection_incomplete(self) -> None:
        facts = self._facts([SegmentState.TYPED_FACT, SegmentState.UNRESOLVED])
        report = facts.completeness()
        assert report.complete is False
        assert report.unresolved == 1
        assert report.unresolved_lines == (2,)
        assert "INCOMPLETE" in report.describe()

    def test_no_unresolved_segment_makes_it_complete(self) -> None:
        facts = self._facts([SegmentState.TYPED_FACT, SegmentState.TYPED_FACT])
        report = facts.completeness()
        assert report.complete is True
        assert report.unresolved == 0
        assert report.unresolved_lines == ()
        assert "INCOMPLETE" not in report.describe()

    def test_the_two_differ_so_the_flag_is_not_a_constant(self) -> None:
        incomplete = self._facts(
            [SegmentState.TYPED_FACT, SegmentState.UNRESOLVED]
        ).completeness()
        complete = self._facts(
            [SegmentState.TYPED_FACT, SegmentState.TYPED_FACT]
        ).completeness()
        assert incomplete.complete != complete.complete
        assert incomplete.describe() != complete.describe()

    def test_a_projection_with_no_account_reads_as_not_established(self) -> None:
        """The fail-closed direction, asserted rather than assumed."""
        from mao.trust.classes import SafeSynthesisContext

        context = SafeSynthesisContext(
            context_id="x", clinical_question="Is donepezil safe?"
        )
        assert context.is_complete() is False
        assert "NOT ESTABLISHED" in context.render()

    def test_an_identifier_only_document_is_not_forced_to_refuse(self) -> None:
        """The A-1 caveat, which the account is supposed to have survived.

        A letterhead with nothing clinical on it has genuinely lost nothing, so
        coverage is 1.0 rather than 0/0. This is the direction the old
        denominator collapsed in, and asserting it keeps the fix honest in both
        directions rather than only the loud one.
        """
        assert CompletenessReport().coverage() == 1.0
        assert CompletenessReport().complete is True
