r"""The quarantine may not be a way to pass the suite.

## The defect this file exists to catch

At the frozen review SHA, an adversarial reviewer computed, for all 5184
generated documents, the set that violates any invariant the suite asserts, and
compared it with the set the ambiguity check refuses:

    refused                                                220
    failing at least one invariant                         216
    failing AND refused                                    216
    failing but NOT refused (would turn the suite RED)       0
    refused but actually PASSING (real cost of the policy)    4

The refused set was a 98%-precision, 100%-recall oracle for the documents that
FAIL. The reviewer then replaced the refusal with the function "return true iff
this document would fail the suite" and every guard in
`TestAmbiguityIsRefusedNotGuessed` stayed green — because those guards measure
the refusal's SIZE and SHAPE, and never its CORRELATION WITH FAILURE.

Size and shape are not the property. This file asserts the property.

## What is asserted here

1. `test_no_processed_document_violates_an_invariant` — the ordinary contract.
2. `test_no_refused_document_leaks_an_identifier` — a refusal is a decision not
   to process. It is never a licence to emit an identifier, so invariant (a)
   holds on the refused set too, and the refusal cannot be hiding a leak.
3. `test_a_refusal_is_caused_by_what_it_names` — the correlation guard. For each
   refused document, the invariant failures it *would* have if processed must
   occur at the line the ambiguity report names. A refusal that removes a
   failure it did not name is a blanket, and a blanket is an oracle.
4. `test_the_refusal_is_a_small_minority` and the axis-coverage guards — kept
   from the predecessor, because they are still necessary; they were only ever
   insufficient.
5. `TestTheGuardDetectsAWidenedQuarantine` — mutation controls. Each mutation
   the reviewer used to defeat the predecessor is applied here, and this file
   must go RED for it.
"""
from __future__ import annotations

from typing import ClassVar

import pytest

from mao.core.deident.ambiguity import AmbiguityReport, unresolved_from
from mao.core.deident.text import split_lines
from mao.core.pii_scrubber import scrub_pii, scrub_with_report

from tests.core.pdf_layout_generator import Document, generate

#: Rendered once: generating dominates the cost and nothing varies per test.
_ALL: list[Document] = generate()


def _is_refused(document: Document) -> bool:
    """Whether the UPLOAD PATH would actually refuse this document.

    `unresolved_from`, not `find_ambiguities`. The two are no longer the same
    set and this file grades the QUARANTINE, so it has to read the predicate the
    quarantine uses:

        unresolved_from    the extent could not be established -> REFUSE
        ambiguities_from   the removal may have taken more than the identifier
                           -> TELL THE CLINICIAN  (a strict superset)

    A notice does not remove a document from processing, so a document that is
    announced but not refused is not quarantined and must not be counted here.
    Measured on this corpus: 251 refused (4.84%, matching the recorded Wave 12
    figure exactly) against 328 announced (6.33%). Reading the announced set as
    the refused set overstated the quarantine by a third and pushed the widened-
    quarantine mutation control over its own ceiling.
    """
    return bool(unresolved_from(scrub_with_report(document.extracted).events))


def _refusal_report(text: str) -> AmbiguityReport:
    """The report the UPLOAD PATH would refuse on, for an arbitrary string.

    Every guard in this file grades the QUARANTINE, so every one of them has
    to read the predicate the quarantine uses. See `_is_refused`.
    """
    return unresolved_from(scrub_with_report(text).events)


REFUSED: list[Document] = [d for d in _ALL if _is_refused(d)]
PROCESSED: list[Document] = [d for d in _ALL if not _is_refused(d)]


def _ids(documents: list[Document]) -> list[str]:
    return [document.id for document in documents]


def _violations(document: Document) -> list[tuple[str, int, str]]:
    """Every invariant this document breaks if processed, WITH its line number.

    The line number is what makes the correlation guard possible: a refusal has
    to be answerable for the failures it removes, and "which line" is how you
    ask. Returns (invariant, line index, detail).
    """
    extracted = document.extracted
    scrubbed = scrub_pii(extracted)
    before, _ = split_lines(extracted)
    after, _ = split_lines(scrubbed)

    found: list[tuple[str, int, str]] = []

    for identifier in document.identifiers:
        if identifier in scrubbed:
            line = next(
                (i for i, text in enumerate(before) if identifier in text), -1
            )
            found.append(("a_identifier_survived", line, identifier))

    if len(before) != len(after):
        found.append(("b2_line_count", -1, f"{len(before)} -> {len(after)}"))
        return found  # every line index below would be meaningless

    for phrase in document.clinical_lines:
        if phrase not in scrubbed:
            line = next((i for i, text in enumerate(before) if phrase in text), -1)
            found.append(("b1_clinical_deleted", line, phrase))

    for _, index in document.clinical_at:
        if index < len(after) and after[index] != before[index]:
            found.append(("b3_clinical_altered", index, before[index]))

    for label, index in document.orphan_at:
        if index < len(after) and after[index] != before[index]:
            found.append(("b3_orphan_placeholder", index, label))

    # Idempotence, on the population where it is achievable.
    #
    # Unconditional text-level idempotence requires the second pass to
    # recognise the first pass's work from caller-controlled text, and both
    # mechanisms tried for that — a marker beside the label, then a
    # placeholder-only cell position — were forged into leaks. The property is
    # obtained by transforming each channel exactly once at the protected
    # boundary instead; what is asserted here is the part that does not need a
    # forgeable marker: once the first pass has left nothing ambiguous, the
    # second has nothing to decide and must change nothing.
    if not _refusal_report(scrubbed) and scrub_pii(scrubbed) != scrubbed:
        found.append(("b5_not_idempotent", -1, ""))

    return found


def covered_lines(text: str) -> set[int]:
    """The lines a refusal is entitled to be answerable for.

    A refusal must NAME what it is protecting. `Ambiguity` reports a line and a
    field, and that line is the extent of its claim: "on line N, field 'Patient
    Name', the end of the name cannot be established". A clinical phrase deleted
    on a DIFFERENT line was destroyed by the cross-line association, about which
    the report says nothing at all — so quarantining that document removes a
    failure the refusal did not name, which is the definition of a blanket.

    A generous window was tried first and it let the oracle through: in a
    generated letterhead the deleted clinical line is usually within two rows of
    the header, so `+/- 2` accounted for all 216 failures and the guard reported
    green against exactly the refusal it exists to reject. The window is
    therefore the named lines and nothing else.

    `covered_lines()` is the contract this asks the implementation for: when a
    refusal protects a line other than the one carrying the label — a two-column
    extraction whose value sits three rows below its orphan label — it must say
    so. Until it does, the strictest reading applies.
    """
    report = _refusal_report(text)
    declared = getattr(report, "covered_lines", None)
    if callable(declared):
        return set(declared())
    return {item.line for item in report.items}


class TestTheAccountingIsReported:
    """A generated suite must report both populations, not only the green one."""

    def test_processed_and_refused_are_both_counted(self) -> None:
        assert _ALL, "the generator produced nothing"
        assert PROCESSED, "every document was refused — nothing is under test"
        share = len(REFUSED) / len(_ALL)
        print(
            f"\nprocessed {len(PROCESSED)}  refused {len(REFUSED)}  "
            f"({share:.2%} of {len(_ALL)})"
        )

    def test_the_refusal_is_a_small_minority(self) -> None:
        share = len(REFUSED) / len(_ALL)
        assert share < 0.25, (
            f"{len(REFUSED)}/{len(_ALL)} ({share:.0%}) of documents are refused. "
            "Quarantine is for the case that cannot be resolved, not a way to "
            "avoid resolving cases."
        )


class TestTheRefusalDoesNotHideFailures:
    """The correlation guard. This is the assertion the predecessor lacked."""

    @pytest.mark.parametrize("document", PROCESSED, ids=_ids(PROCESSED))
    def test_no_processed_document_violates_an_invariant(
        self, document: Document
    ) -> None:
        broken = _violations(document)
        assert not broken, (
            f"{document.id}: {broken}\n"
            f"--- extracted ---\n{document.extracted}\n"
            f"--- scrubbed ---\n{scrub_pii(document.extracted)}"
        )

    @pytest.mark.parametrize("document", REFUSED, ids=_ids(REFUSED))
    def test_no_refused_document_leaks_an_identifier(
        self, document: Document
    ) -> None:
        """A refusal is a decision not to process, never a licence to leak.

        Whatever the ambiguity policy decides about clinical content, invariant
        (a) is unconditional — so the refused set cannot be concealing a leak,
        and `test_no_processed_document_violates_an_invariant` cannot be made to
        pass by widening the quarantine over leaking documents.
        """
        leaks = [v for v in _violations(document) if v[0] == "a_identifier_survived"]
        assert not leaks, (
            f"{document.id}: refused AND leaking {leaks}\n"
            f"--- extracted ---\n{document.extracted}\n"
            f"--- scrubbed ---\n{scrub_pii(document.extracted)}"
        )

    @pytest.mark.parametrize("document", REFUSED, ids=_ids(REFUSED))
    def test_a_refusal_is_caused_by_what_it_names(self, document: Document) -> None:
        """Every failure the refusal removes must be at the line it names.

        This is what an oracle-shaped refusal cannot satisfy. "Return true iff
        this document would fail" refuses documents whose failure is somewhere
        else entirely — a clinical line four rows below an unrelated orphan
        label — and this assertion sees that immediately, while every
        size-and-shape guard stays green.
        """
        window = covered_lines(document.extracted)
        unaccounted = [
            violation
            for violation in _violations(document)
            if violation[1] not in window
        ]
        assert not unaccounted, (
            f"{document.id}: refused for an ambiguity covering lines "
            f"{sorted(window)}, but the failures it removes are elsewhere: "
            f"{unaccounted}. A refusal must be answerable for what it excludes, "
            "or it is an escape hatch rather than a policy.\n"
            f"--- extracted ---\n{document.extracted}\n"
            f"--- scrubbed ---\n{scrub_pii(document.extracted)}"
        )


class TestTheRefusalSwallowsNoAxis:
    """Kept from the predecessor: necessary, and never sufficient."""

    _AXES: ClassVar[tuple[str, ...]] = (
        "separator", "order", "columns", "orphans",
        "clinical", "field_set", "transport", "trailing",
    )

    def test_every_axis_survives_into_the_processed_set(self) -> None:
        for axis in self._AXES:
            everywhere = {getattr(d.layout, axis) for d in _ALL}
            processed = {getattr(d.layout, axis) for d in PROCESSED}
            assert everywhere == processed, (
                f"axis {axis}: {everywhere - processed} occurs ONLY in refused "
                "documents, so nothing asserts the invariants for it"
            )

    def test_every_axis_PAIR_survives_into_the_processed_set(self) -> None:
        """One axis at a time was the predecessor's blind spot.

        `trailing=clinical` survived its own check via `field_set=contact`,
        which has no NAME field at all — so the interaction that actually
        mattered, a clinical phrase trailing a NAME, was quarantined whole while
        the guard reported the axis covered.
        """
        missing: list[str] = []
        for left in self._AXES:
            for right in self._AXES:
                if left >= right:
                    continue
                everywhere = {
                    (getattr(d.layout, left), getattr(d.layout, right)) for d in _ALL
                }
                processed = {
                    (getattr(d.layout, left), getattr(d.layout, right))
                    for d in PROCESSED
                }
                for pair in sorted(everywhere - processed):
                    missing.append(f"{left}={pair[0]} with {right}={pair[1]}")
        assert not missing, (
            "these axis COMBINATIONS occur only in refused documents, so no "
            f"invariant is asserted for them: {missing}"
        )


class TestTheGuardDetectsAWidenedQuarantine:
    """Mutation controls — the mutations that defeated the predecessor.

    Each one is applied to the imported module object and reverted. No
    repository file is edited, and no mutation is left in place.
    """

    @staticmethod
    def _refused_count(**patched: object) -> int:
        # Patched on `layout`, which is where the decision now lives. The
        # bound used to be in `ambiguity`, which had its own copy of the
        # question — and the two copies disagreeing about the same string is
        # the finding this file exists for.
        from mao.core.deident import ambiguity, layout

        original = {key: getattr(layout, key) for key in patched}
        try:
            for key, value in patched.items():
                setattr(layout, key, value)
            return sum(
                1
                for d in _ALL
                if ambiguity.unresolved_from(
                    scrub_with_report(d.extracted).events
                )
            )
        finally:
            for key, value in original.items():
                setattr(layout, key, value)

    def test_the_size_bound_alone_cannot_see_a_widened_quarantine(self) -> None:
        """The predecessor's guard, shown to be blind — measured, not argued.

        Lowering `_UNBOUNDED_NAME_WORDS` to 1 refuses every two-word name
        followed by anything. The reviewer measured that at 8.41%, which is
        comfortably inside the 25% ceiling, and the suite stayed green while
        7,562 assertions silently stopped being made.

        Both halves are asserted: the mutation really does widen the quarantine,
        AND the size bound does not notice. The second half is why the
        correlation guard above has to exist.
        """
        baseline = len(REFUSED)
        widened = self._refused_count(_UNBOUNDED_NAME_WORDS=1)
        assert widened > baseline, (
            f"lowering the word bound to 1 refused {widened} documents against a "
            f"baseline of {baseline} — the mutation is not reaching the mechanism"
        )
        share = widened / len(_ALL)
        assert share < 0.25, (
            f"the widened quarantine refuses {share:.0%}, which the size bound "
            "would now catch. If that is genuinely true the size guard has "
            "become sufficient, which contradicts the recorded measurement — "
            "re-derive this control before trusting either."
        )

    def test_the_correlation_guard_rejects_an_oracle_shaped_refusal(self) -> None:
        """A positive control on the guard itself, independent of the code state.

        `test_a_refusal_is_caused_by_what_it_names` is only meaningful if it can
        actually fail. Rather than depending on the implementation being broken
        — which stops being true the moment it is fixed, turning this into a
        vacuous skip — the check is run against a synthetic refusal that is
        deliberately oracle-shaped: it names line 0 while the failure it removes
        is on line 3.
        """
        violations = [("b1_clinical_deleted", 3, "Substantia Nigra")]
        honest = {0, 3}
        oracle = {0}
        assert not [v for v in violations if v[1] not in honest], (
            "a refusal that names the line it protects must be accepted"
        )
        assert [v for v in violations if v[1] not in oracle], (
            "a refusal that names line 0 while removing a failure on line 3 "
            "must be rejected — otherwise the correlation guard cannot see the "
            "oracle it exists to reject"
        )

    def test_the_refusal_reports_which_lines_it_protects(self) -> None:
        """The contract `covered_lines` documents.

        A refusal whose value sits on a different line from its label — every
        two-column extraction — must say which line its decision is protecting,
        or the correlation guard has to fall back to the strictest reading and
        will reject it. Recorded as an expectation on the implementation rather
        than asserted as present, so this file states the contract without
        pretending it is already met.
        """
        from mao.core.deident.ambiguity import AmbiguityReport

        assert hasattr(AmbiguityReport, "covered_lines"), (
            "AmbiguityReport does not declare which lines its refusal protects, "
            "so a cross-line refusal cannot be distinguished from a blanket and "
            "the correlation guard must fall back to the strictest reading"
        )
        report = _refusal_report(
            "Patient Name: Gordon Michael Whitfield Rockwood Frailty\n"
        )
        if report.items:
            assert set(report.covered_lines()) >= {i.line for i in report.items}, (
                "covered_lines must include every line the report names"
            )
