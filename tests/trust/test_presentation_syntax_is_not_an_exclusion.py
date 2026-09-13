r"""Caller-controlled presentation syntax is not an accounting decision.

## The contract

`SourceAccounting` attributes every part of the protected text to exactly one
state, and only two things may put a segment in `STRUCTURAL`:

    a span the TRANSFORMATION recorded  — the field label that attributed a
                                          removal, carried on the event itself;
    a run with no meaning to lose       — no alphanumeric character at all.

Both are facts about what happened to the document. Neither is a fact about
what the document LOOKS like. The rule this file binds:

    Caller-controlled presentation syntax — `#`, `##`, markdown, bullets, list
    markers, punctuation, whitespace and layout, fracture shorthand, whatever a
    PDF extractor happens to emit — must NEVER, BY ITSELF, remove meaning-bearing
    source residue from the accountable population.

and the load-bearing invariant underneath it:

    For arbitrary meaning-bearing source residue, EITHER it is represented in
    the safe projection, OR the projection reports itself INCOMPLETE and
    identifies the relevant source location.

    `absent && complete` must be impossible.

## The defect this exists to make unreachable

`AR18-1` / `ADV18-1`, found independently by both `46a198a` reviewers as the
sole blocker. `accounting.py` carried a `_MARKDOWN_HEADING = re.compile(r"^\s*#+\s")`
branch that attributed a whole line's meaning-bearing residue to `STRUCTURAL`
whatever it said — placing it in neither the numerator nor the denominator of
`coverage()`, out of `unresolved`, and out of `unresolved_lines`. Measured at
`46a198a`: `# Permanent pacemaker in situ` absent from the payload handed to a
model asked whether a bradycardic drug was safe, at `coverage 1.0000`, with the
payload stating `Complete`. `#` is the standard UK shorthand for FRACTURE and
the standard problem-list marker, so this is ordinary clinical notation.

It was the FOURTH predicate in this position across four gates — `_is_clinical_line`
(lexicon), `_carries_nothing_to_lose` (the colon), `_PLACEHOLDER` (bracketed
uppercase), `_MARKDOWN_HEADING` (the hash). `00_RULES.md`'s escalation clause
forbids a fifth, so the decision point was REMOVED rather than answered better.

## How this oracle is independent of the fix it guards

This file would fail identically against a `_MARKDOWN_HEADING` branch, against a
bullet branch, against a "but not if it looks clinical" tie-break, and against
any future exclusion keyed on the appearance of source text. It does not know
which branch was deleted and never mentions one:

  the properties are METAMORPHIC and ABSOLUTE, not expected values
      P1 asserts a disjunction that must hold of every document. P2 compares a
      document against ITSELF with one presentation prefix added and asserts the
      accountable population never shrinks. Neither encodes what the account
      should say, only what it may not do.

  the corpus is GENERATED, not a phrase list
      `_statements()` composes arbitrary meaning-bearing text from parts,
      including deliberate nonsense. Nothing here is a clinical vocabulary, and
      the property holds for a statement the projection carries (branch a) and
      one it does not (branch b) alike, so the corpus cannot be tuned.

  "meaning-bearing" and the source line are derived HERE
      `_carries_meaning` is restated locally and `_source_line_of` counts
      separators in the raw document. The oracle never asks `accounting`,
      `split_lines` or `CompletenessReport` where a line is or what counts.

  the mutation controls fire on the CLASS, not on one branch
      `TestTheOracleFires` restores a heading exclusion AND an unrelated bullet
      exclusion by monkeypatching the state machine, and demands that both
      properties fail. An oracle that only detected the exact branch that was
      removed would pass the first and fail the second.

  the boundary is the real one
      `protect_channel(..., InputChannel.REPORT, ...)` then
      `compile_handoff(..., events=...)` — the calls `clinical_agent` makes.
      Imports are from `mao/` and the standard library only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

import pytest

from mao.core.deident.text import LINE_BREAKS
from mao.trust.classes import InputChannel
from mao.trust.handoff import accounting as accounting_module
from mao.trust.handoff.accounting import SegmentState
from mao.trust.handoff.compiler import HandoffRefused, compile_handoff
from mao.trust.inputs.boundary import protect_channel

QUESTION = "Is donepezil safe for this patient?"

#: A header that makes the boundary actually redact, so the account is exercised
#: with real events rather than an empty record.
HEADER_LINES = ("Patient Name: Harold Nkemdirim", "MRN: RGT/44219/B")

#: Every separator the package supports, named ONCE by the package.
SEPARATORS: tuple[str, ...] = ("\r\n", *LINE_BREAKS)

#: Presentation syntax a caller, a markdown editor or a PDF extractor controls.
#: None of it is a statement about meaning; all of it is a statement about
#: layout. The list mixes shapes the deleted branch matched with shapes it did
#: not, deliberately, so the property is about the CLASS and not about `#`.
PRESENTATION_PREFIXES: tuple[str, ...] = (
    "# ",
    "## ",
    "### ",
    "#\t",
    "   # ",
    "#  ",
    "- ",
    "* ",
    "> ",
    "1. ",
    "• ",
    "\t",
)


def _carries_meaning(text: str) -> bool:
    """The contract's own definition of meaning-bearing, restated locally.

    Deliberately not imported from `accounting`: an oracle that asks the module
    under test what counts as meaning cannot detect the module redefining it.
    """
    return any(character.isalnum() for character in text)


def _statements() -> tuple[str, ...]:
    """Arbitrary meaning-bearing residue, composed rather than curated.

    NOT a phrase whitelist and not a clinical lexicon. Three of the four parts
    are ordinary administrative or nonsense prose, and the property under test
    is satisfied either by carriage or by an honest incompleteness report, so a
    statement being clinically interesting is irrelevant to whether it must
    hold.

    Shapes whose meaning is deliberately carried under a DIFFERENT spelling —
    an age, a vital, a lab value — are excluded, because for those the payload
    legitimately contains `older_adult_75_89` rather than `Aged 84`, and a
    substring test would read that transformation as a loss. They are the one
    population where "absent from the payload" is not evidence of anything.
    """
    subjects = ("Permanent pacemaker", "Qualtrough review", "NOF", "Widget fettling")
    tails = (
        "in situ",
        "documented 2020",
        "declined on advice",
        "to be monitored monthly",
    )
    return tuple(f"{subject} {tail}" for subject in subjects for tail in tails)


def _document(statement: str, prefix: str, separator: str) -> str:
    """One letter carrying `statement` on a known line, at a known prefix."""
    body = (*HEADER_LINES, f"{prefix}{statement}", "Donepezil 10mg od")
    return separator.join(body) + separator


def _source_line_of(document: str, statement: str, separator: str) -> int:
    """The 1-based line the statement sits on, derived by counting separators.

    Independent of `split_lines` and of anything the account reports.
    """
    for number, line in enumerate(document.split(separator), start=1):
        if statement in line:
            return number
    raise AssertionError(f"{statement!r} is not in the document")


@dataclass(frozen=True)
class Verdict:
    """What the product did with one document."""

    refused: bool
    represented: bool
    complete: bool
    unresolved_lines: tuple[int, ...]
    accountable: int


def _verdict(document: str, statement: str) -> Verdict:
    """Drive the real boundary and the real compiler and read the outcome.

    A refusal is a permitted outcome and is recorded as one: the projection that
    refuses has made no claim about completeness, so it cannot make a false one.
    """
    safe = protect_channel(document, InputChannel.REPORT, refuse_ambiguity=False)
    try:
        handoff = compile_handoff(safe.text, question=QUESTION, events=safe.events)
    except HandoffRefused:
        return Verdict(True, False, False, (), 0)
    report = handoff.facts.completeness()
    return Verdict(
        refused=False,
        represented=statement.lower() in handoff.synthesis_context.render().lower(),
        complete=report.complete,
        unresolved_lines=report.unresolved_lines,
        accountable=report.accountable,
    )


def _silently_lost(verdict: Verdict) -> bool:
    """`absent && complete` — the conjunction the contract forbids outright."""
    return (not verdict.refused) and (not verdict.represented) and verdict.complete


# --------------------------------------------------------------------------
# P1 — the disjunction, absolutely
# --------------------------------------------------------------------------


class TestMeaningBearingResidueIsNeverBothAbsentAndComplete:
    """Either represented, or reported incomplete with the location named."""

    @pytest.mark.parametrize("statement", _statements())
    @pytest.mark.parametrize("prefix", PRESENTATION_PREFIXES)
    def test_no_presentation_prefix_hides_a_statement(
        self, statement: str, prefix: str
    ) -> None:
        document = _document(statement, prefix, "\n")
        verdict = _verdict(document, statement)
        assert not _silently_lost(verdict), (
            f"{prefix + statement!r} is ABSENT from the payload while the "
            f"projection reports complete={verdict.complete}. Meaning-bearing "
            "residue left the accountable population because of how it was "
            "spelled, which is the AR18-1 / ADV18-1 defect."
        )

    @pytest.mark.parametrize("separator", SEPARATORS)
    def test_no_supported_line_separator_hides_a_statement(
        self, separator: str
    ) -> None:
        statement = "Permanent pacemaker in situ"
        for prefix in PRESENTATION_PREFIXES:
            document = _document(statement, prefix, separator)
            verdict = _verdict(document, statement)
            assert not _silently_lost(verdict), (
                f"{prefix + statement!r} separated by {separator!r} is absent "
                "from the payload and the projection claims to be complete."
            )

    @pytest.mark.parametrize("prefix", PRESENTATION_PREFIXES)
    def test_an_unrepresented_statement_has_its_source_line_named(
        self, prefix: str
    ) -> None:
        """The second half: incomplete is not enough, the location must be right.

        A clinician told "something is missing" and pointed at the wrong line
        cannot act on it. The expected line is counted from the raw document.
        """
        statement = "Qualtrough review to be monitored monthly"
        document = _document(statement, prefix, "\n")
        verdict = _verdict(document, statement)
        if verdict.refused or verdict.represented:
            return
        expected = _source_line_of(document, statement, "\n")
        assert expected in verdict.unresolved_lines, (
            f"{prefix + statement!r} is on source line {expected} and is not "
            f"carried, but the projection names {verdict.unresolved_lines}."
        )


# --------------------------------------------------------------------------
# P2 — the metamorphic property: presentation syntax changes nothing
# --------------------------------------------------------------------------


class TestPresentationSyntaxNeverShrinksTheAccountablePopulation:
    """The same text, one prefix apart, must not be accounted for less.

    This is the one-character isolating control both `46a198a` reviewers used,
    generalised into a property. It holds whether or not the statement is
    carried, so it detects an exclusion even on text the projection represents —
    which a disjunction test alone cannot do.
    """

    @pytest.mark.parametrize("statement", _statements())
    @pytest.mark.parametrize("prefix", PRESENTATION_PREFIXES)
    def test_adding_a_prefix_does_not_reduce_what_is_accounted(
        self, statement: str, prefix: str
    ) -> None:
        plain = _verdict(_document(statement, "", "\n"), statement)
        prefixed = _verdict(_document(statement, prefix, "\n"), statement)
        if plain.refused or prefixed.refused:
            return
        assert prefixed.accountable >= plain.accountable, (
            f"adding {prefix!r} took {plain.accountable - prefixed.accountable} "
            f"segment(s) out of the accountable population for {statement!r}. "
            "Presentation syntax decided what gets measured."
        )

    @pytest.mark.parametrize("statement", _statements())
    @pytest.mark.parametrize("prefix", PRESENTATION_PREFIXES)
    def test_a_prefix_cannot_turn_an_honest_outcome_into_a_silent_one(
        self, statement: str, prefix: str
    ) -> None:
        plain = _verdict(_document(statement, "", "\n"), statement)
        prefixed = _verdict(_document(statement, prefix, "\n"), statement)
        if _silently_lost(plain):
            return
        assert not _silently_lost(prefixed), (
            f"{statement!r} is accounted honestly on its own and silently lost "
            f"once {prefix!r} is in front of it."
        )


# --------------------------------------------------------------------------
# The exact predecessor reproductions, verbatim from the frozen reports
# --------------------------------------------------------------------------


class TestTheReviewedDocumentsReproduce:
    """AR18-1's and ADV18-1's own measured inputs, quoted rather than restated."""

    #: Named in the two reports as the shapes that fired. Reproductions, not the
    #: corpus the properties above are graded on.
    REPORTED = (
        "# Permanent pacemaker in situ",
        "# Complete Heart Block",
        "# Penicillin anaphylaxis documented 2020",
        "# NOF",
        "# NOF, ORIF 2019, weight bearing as tolerated",
        "# ribs 4-6 left",
        "## Problem list",
        "## Findings",
        "# Department of Cardiology opinion sought",
        "# GP to monitor pulse monthly",
        "# Surgery declined on cardiology advice",
    )

    @pytest.mark.parametrize("line", REPORTED)
    def test_the_reported_line_is_not_silently_lost(self, line: str) -> None:
        statement = line.lstrip("#").lstrip()
        document = _document(statement, line[: len(line) - len(statement)], "\n")
        assert not _silently_lost(_verdict(document, statement)), (
            f"{line!r} reproduces AR18-1 / ADV18-1: absent from the payload "
            "while the payload asserts completeness."
        )

    def test_adv18_1_measurement_1_the_isolating_control(self) -> None:
        """Removing one character must not change whether the loss is announced."""
        statement = "Permanent pacemaker in situ"
        with_hash = _verdict(_document(statement, "# ", "\n"), statement)
        control = _verdict(_document(statement, "", "\n"), statement)
        assert _silently_lost(with_hash) == _silently_lost(control) is False
        assert with_hash.accountable == control.accountable

    def test_adv18_1_measurement_3_the_named_line_is_the_right_one(self) -> None:
        """The drift case: the report must not point at the letterhead instead.

        Measured at `46a198a`: reported "INCOMPLETE: 1 of 5 ... (source line 1)"
        — the letterhead — while the pacemaker on line 6 was `STRUCTURAL` and
        was never named.
        """
        document = (
            "Leeds Teaching Hospitals NHS Trust\n"
            "Patient Name: Harold Nkemdirim\n"
            "MRN: RGT/44219/B\n"
            "## Problem list\n"
            "# Complete heart block\n"
            "# Permanent pacemaker in situ (Medtronic Azure, 2021)\n"
            "# Penicillin allergy - anaphylaxis\n"
            "Current medications\n"
            "Donepezil 10mg od\n"
        )
        statement = "Permanent pacemaker in situ (Medtronic Azure, 2021)"
        verdict = _verdict(document, statement)
        assert not _silently_lost(verdict)
        if not verdict.represented and not verdict.refused:
            expected = _source_line_of(document, statement, "\n")
            assert expected in verdict.unresolved_lines, (
                f"the pacemaker is on line {expected}; the clinician is pointed "
                f"at {verdict.unresolved_lines}."
            )


# --------------------------------------------------------------------------
# Non-vacuity: the oracle fires when the exclusion CLASS is restored
# --------------------------------------------------------------------------


def _exclusion_mutant(pattern: str):
    """A state machine that re-excludes residue by how the source LOOKS.

    Wraps the real `_account_line` and re-labels whatever it accounted
    `UNRESOLVED` as `STRUCTURAL` when the line matches `pattern`. That is the
    removed branch's exact effect, expressed without copying its code, so it
    stands in for ANY exclusion keyed on presentation rather than provenance.
    """
    compiled = re.compile(pattern)
    real = accounting_module._account_line

    def mutant(index, base, line, marks):  # type: ignore[no-untyped-def]
        segments = real(index, base, line, marks)
        if not compiled.match(line):
            return segments
        return [
            replace(segment, state=SegmentState.STRUCTURAL)
            if segment.state is SegmentState.UNRESOLVED
            else segment
            for segment in segments
        ]

    return mutant


class TestTheOracleFires:
    """Mutation controls. An oracle that cannot fail is not evidence.

    Both mutants are restorations of the FORBIDDEN CLASS rather than of one
    deleted branch: the first is the heading exclusion AR18-1 named, the second
    is an exclusion on a prefix that branch never matched. The properties above
    must reject both, or they are guarding an implementation detail instead of
    the contract.
    """

    MUTANTS = (
        pytest.param(r"^\s*#+\s", "# ", id="heading-exclusion-restored"),
        pytest.param(r"^\s*[-*•]\s", "- ", id="bullet-exclusion-restored"),
    )

    @pytest.mark.parametrize("pattern, prefix", MUTANTS)
    def test_p1_rejects_a_restored_presentation_exclusion(
        self, monkeypatch: pytest.MonkeyPatch, pattern: str, prefix: str
    ) -> None:
        monkeypatch.setattr(
            accounting_module, "_account_line", _exclusion_mutant(pattern)
        )
        statement = "Permanent pacemaker in situ"
        verdict = _verdict(_document(statement, prefix, "\n"), statement)
        assert _silently_lost(verdict), (
            "the mutant restores an exclusion keyed on presentation and P1 did "
            "not notice — the disjunction test is vacuous."
        )

    @pytest.mark.parametrize("pattern, prefix", MUTANTS)
    def test_p2_rejects_a_restored_presentation_exclusion(
        self, monkeypatch: pytest.MonkeyPatch, pattern: str, prefix: str
    ) -> None:
        statement = "Permanent pacemaker in situ"
        plain = _verdict(_document(statement, "", "\n"), statement)
        monkeypatch.setattr(
            accounting_module, "_account_line", _exclusion_mutant(pattern)
        )
        prefixed = _verdict(_document(statement, prefix, "\n"), statement)
        assert prefixed.accountable < plain.accountable, (
            "the mutant takes the statement out of the accountable population "
            "and P2 did not notice — the metamorphic test is vacuous."
        )

    def test_the_mutant_is_the_only_thing_that_changed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A line the mutant's pattern does not match must be unaffected.

        Without this the mutation controls could pass by breaking everything.
        """
        statement = "Permanent pacemaker in situ"
        before = _verdict(_document(statement, "", "\n"), statement)
        monkeypatch.setattr(
            accounting_module, "_account_line", _exclusion_mutant(r"^\s*#+\s")
        )
        after = _verdict(_document(statement, "", "\n"), statement)
        assert before == after


# --------------------------------------------------------------------------
# Only provenance may produce STRUCTURAL
# --------------------------------------------------------------------------


class TestStructuralIsAttributableToTheTransformation:
    """Every `STRUCTURAL` segment is a recorded removal's label, or says nothing.

    This is the audit `ADV18-1` performed by hand, bound as an assertion. It
    closes the class rather than the instance: any future branch that attributes
    meaning-bearing text to `STRUCTURAL` on any grounds other than the
    transformation's own record fails here, whatever it keys on.
    """

    DOCUMENTS = tuple(
        _document(statement, prefix, "\n")
        for statement in _statements()
        for prefix in PRESENTATION_PREFIXES
    )

    @pytest.mark.parametrize("document", DOCUMENTS)
    def test_no_structural_segment_carries_unattributed_meaning(
        self, document: str
    ) -> None:
        safe = protect_channel(document, InputChannel.REPORT, refuse_ambiguity=False)
        try:
            handoff = compile_handoff(
                safe.text, question=QUESTION, events=safe.events
            )
        except HandoffRefused:
            return
        labels = {
            (event.label_start, event.label_end)
            for event in safe.events
            if event.has_label()
        }
        for segment in handoff.facts.accounting.segments:
            if segment.state is not SegmentState.STRUCTURAL:
                continue
            if not _carries_meaning(segment.text):
                continue
            assert (segment.start, segment.end) in labels, (
                f"{segment.text!r} is meaning-bearing and was accounted "
                "STRUCTURAL without a redaction event attributing it. "
                "STRUCTURAL is a provenance claim, not an appearance claim."
            )


def test_the_accounting_module_holds_no_source_text_predicate() -> None:
    """No regex over the appearance of source text may live in the account.

    A structural guard, not a style rule. Four consecutive gates failed on a
    predicate in exactly this position, and each was introduced as a small,
    obviously-correct special case. The module needs no pattern over source text
    to do its job: it reads the transformation's event record and the spans the
    projection carried, both of which are facts about what HAPPENED.
    """
    patterns = [
        name
        for name, value in vars(accounting_module).items()
        if isinstance(value, re.Pattern)
    ]
    assert patterns == [], (
        f"{patterns} match against source text inside the account. Whatever "
        "this one is for, it is the fifth predicate 00_RULES forbids."
    )
