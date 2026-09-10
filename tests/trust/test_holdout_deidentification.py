r"""The two de-identification invariants, on a corpus the implementation has
never been fitted to.

`tests/core/test_pii_scrubber_layouts.py` asserts the same two invariants on a
generated population whose WORDS were chosen to satisfy the mechanism under
test. This file asserts them on `tests/trust/holdout.py`, whose words were not,
and `TestTheCorpusIsGenuinelyHeldOut` is what keeps that true over time.

The two directions, and why both are asserted on the same documents:

  (a) no identifier is readable in the output — under-matching leaks;
  (b) no clinical content is deleted — over-matching destroys, silently, on a
      path whose prompt tells the model the text "has already been
      de-identified".

Five remediation rounds closed one direction and reopened the other. Asserting
them separately, on separate corpora, is how that kept being possible.
"""
from __future__ import annotations

import unicodedata
from typing import ClassVar

import pytest

from mao.core.pii_scrubber import scrub_pii

from .holdout import (
    HOLDOUT_CLINICAL,
    INVISIBLE_CHARACTERS,
    Case,
    all_cases,
    clinical_preservation_documents,
    combined_documents,
    decorated_value_documents,
    forged_placeholder_documents,
    invisible_character_documents,
    middle_name_documents,
    name_boundary_documents,
)

_CASES: list[Case] = all_cases()


def _ids(cases: list[Case]) -> list[str]:
    return [case.id for case in cases]


def visible(text: str) -> str:
    """The text with everything that occupies no advance width removed.

    This is what a reader, a log, and a third-party model actually see. An
    assertion written against the raw string calls
    `Patient Name: [NAME]<U+034F>dirim Okonkwo` a pass, because `Nkemdirim` is
    not a literal substring of it — and that output is strictly worse than no
    redaction at all, since it asserts a removal it did not perform.

    Deliberately implemented HERE and not imported from `mao.core.deident.text`.
    A test that measures the scrubber with the scrubber's own definition of
    "invisible" cannot see a character the scrubber's definition is missing,
    which is precisely how the previous residue survived a fix that closed 15 of
    21 characters.
    """
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text)
        if not _is_zero_width(character)
    )


def _is_zero_width(character: str) -> bool:
    category = unicodedata.category(character)
    if category in ("Cf", "Mn", "Me", "Cc"):
        return character not in "\t\n\r"
    # Hangul fillers (`Lo`) and the Braille blank (`So`) render as nothing.
    return character in "ᅟᅠㅤﾠ⠀"


class TestTheCorpusIsGenuinelyHeldOut:
    """The corpus must stay OUTSIDE the implementation's vocabulary.

    This is the assertion the whole file rests on. `pdf_layout_generator`'s own
    docstring admits its trailing corpus was "chosen to END in a clinical head
    noun, which is the signal the boundary rule relies on" — a corpus selected
    to satisfy the mechanism it measures. 49 of 60 phrases from outside that
    selection were then destroyed with no refusal at all.

    It also fails if somebody closes a hold-out failure by adding these words to
    `lexicon.py`. That is the "open-ended vocabulary/regex patch cycle" the
    control documents forbid, and this test is what makes taking it visible.
    """

    def test_most_clinical_phrases_are_unknown_to_the_lexicon(self) -> None:
        from mao.core.deident.lexicon import NOT_A_NAME, has_clinical_head
        from mao.core.deident.values import _fold, _parts

        known = []
        for phrase in HOLDOUT_CLINICAL:
            words = phrase.split()
            in_lexicon = any(
                part in NOT_A_NAME for word in words for part in _parts(word) if part
            )
            has_head = has_clinical_head([_fold(word) for word in words])
            if in_lexicon or has_head:
                known.append(phrase)

        share = len(known) / len(HOLDOUT_CLINICAL)
        assert share < 0.35, (
            f"{len(known)}/{len(HOLDOUT_CLINICAL)} ({share:.0%}) of the hold-out "
            f"clinical corpus is now inside the implementation's own vocabulary: "
            f"{known}.\n"
            "Either the corpus has drifted towards the lexicon, or a failure was "
            "closed by adding words to it. Both make this file measure the "
            "mechanism against itself, which is the failure it exists to prevent."
        )

    def test_the_corpus_does_not_overlap_the_generator_fixtures(self) -> None:
        """Overlap would re-measure what `tests/core` already measures."""
        from tests.core.pdf_layout_generator import CLINICAL_CORPUS, TRAILING_CLINICAL

        overlap = set(HOLDOUT_CLINICAL) & (set(CLINICAL_CORPUS) | set(TRAILING_CLINICAL))
        assert not overlap, f"hold-out phrases also used as fixtures: {sorted(overlap)}"

    def test_the_population_is_broad_and_every_axis_is_exercised(self) -> None:
        assert len(_CASES) > 500, f"only {len(_CASES)} hold-out documents"
        axes = {axis for case in _CASES for axis in case.axes}
        for required in (
            "name_boundary",
            "middle_name",
            "decorated_value",
            "forged_placeholder",
            "invisible_character",
            "clinical_preservation",
            "combined",
        ):
            assert required in axes, f"axis {required} is not exercised"


class TestNoIdentifierIsReadable:
    """Invariant (a), on the hold-out corpus.

    Asserted on EVERY case including the ones a refusal would quarantine: a
    refusal is a decision not to process, and it can never be a licence to emit
    an identifier. Whatever the ambiguity policy decides, this must hold.
    """

    @pytest.mark.parametrize("case", _CASES, ids=_ids(_CASES))
    def test_no_identifier_survives(self, case: Case) -> None:
        seen = visible(scrub_pii(case.text))
        survivors = [
            identifier
            for identifier in case.must_not_survive
            if visible(identifier) in seen
        ]
        assert not survivors, (
            f"{case.id}: {survivors} are still readable\n"
            f"--- in  ---\n{case.text!r}\n--- out ---\n{seen!r}"
        )


class TestNoPlaceholderCoversPartOfAnIdentifier:
    """The worst outcome available: the name leaks AND the output claims it did not.

    `values.py` asserts this about itself in a comment — "A NAME VALUE IS NEVER
    TRUNCATED TO A PREFIX" — and three separate mechanisms have falsified it: a
    token cap, an eponym at the end of the line, and a lexicon word in the
    middle of the name. Stated here as an executable property rather than a
    comment.
    """

    _PARTIAL: ClassVar[list[Case]] = (
        middle_name_documents() + name_boundary_documents()
    )

    @pytest.mark.parametrize("case", _PARTIAL, ids=_ids(_PARTIAL))
    def test_a_placeholder_is_never_emitted_beside_a_legible_fragment(
        self, case: Case
    ) -> None:
        seen = visible(scrub_pii(case.text))
        if "[NAME]" not in seen:
            return  # nothing was claimed; invariant (a) is what covers this case
        legible = [
            fragment
            for fragment in case.must_not_survive
            if visible(fragment) in seen
        ]
        assert not legible, (
            f"{case.id}: a [NAME] placeholder asserts a removal that did not "
            f"happen — {legible} are still readable beside it\n"
            f"--- in  ---\n{case.text!r}\n--- out ---\n{seen!r}"
        )


class TestNoClinicalContentIsDestroyed:
    """Invariant (b), on phrases the lexicon does not contain.

    A destroyed device name, ward, trial arm or instrument is silently absent
    from the evidence the model reasons over, and the prompt that used to
    accompany it told the model the report "has already been de-identified" —
    so nothing downstream could tell that anything was missing.

    ## The contract has two permitted outcomes, and that is the policy

    `05_CASE_TO_EVIDENCE_SPEC.md` and `ambiguity.py` define one policy with two
    readings, because the two paths differ in who can see a wrong answer:

        upload  the document is processed UNSEEN, so a wrong guess is silent
                either way it goes wrong -> REFUSE, and ask for structured
                patient fields
        chat    the clinician wrote the text and reads the answer, so an
                over-redaction is visible and recoverable while a leak is not
                -> REDACT

    So the assertion is: the phrase survives, OR the document is refused.
    `Rockwood Frailty` under an orphan `Patient Name:` genuinely cannot be told
    from `Harold Nkemdirim` under an orphan `Patient Name:` — the finding this
    corpus exists for is not that the phrase was redacted, it is that it was
    redacted SILENTLY, with no refusal, on the path that promises one.

    `test_the_refusal_cannot_swallow_the_corpus` is what stops this being an
    escape hatch: an implementation that refuses everything fails it.
    """

    _PRESERVING: ClassVar[list[Case]] = [c for c in _CASES if c.must_survive]

    @pytest.mark.parametrize("case", _PRESERVING, ids=_ids(_PRESERVING))
    def test_clinical_content_survives_or_the_document_is_refused(
        self, case: Case
    ) -> None:
        from mao.core.deident.ambiguity import find_ambiguities

        scrubbed = scrub_pii(case.text)
        lost = [phrase for phrase in case.must_survive if phrase not in scrubbed]
        if not lost:
            return
        report = find_ambiguities(case.text)
        assert report, (
            f"{case.id}: clinical content DELETED with NO refusal: {lost}\n"
            f"--- in  ---\n{case.text!r}\n--- out ---\n{scrubbed!r}"
        )
        # ...and the refusal has to be answerable for the line it removed.
        for phrase in lost:
            line = next(
                (
                    index
                    for index, text in enumerate(case.text.splitlines())
                    if phrase in text
                ),
                -1,
            )
            assert line in report.covered_lines(), (
                f"{case.id}: refused, but the refusal names lines "
                f"{sorted(report.covered_lines())} while the content it deleted "
                f"is on line {line}. A refusal that is not answerable for what "
                "it removes is a blanket."
            )

    def test_the_refusal_cannot_swallow_the_corpus(self) -> None:
        """The bound that stops "refuse everything" being a passing strategy."""
        from mao.core.deident.ambiguity import find_ambiguities

        refused = [case for case in _CASES if find_ambiguities(case.text)]
        share = len(refused) / len(_CASES)
        assert share < 0.35, (
            f"{len(refused)}/{len(_CASES)} ({share:.0%}) of the hold-out corpus "
            "is refused. A quarantine that large is not a policy, it is a way to "
            "avoid resolving cases — and this corpus is deliberately weighted "
            "towards the hard ones, so the bound is looser than the generated "
            "suite's 25% rather than tighter."
        )

    def test_an_ordinary_letterhead_is_never_refused(self) -> None:
        """The shape almost every real document has must go straight through."""
        from mao.core.deident.ambiguity import find_ambiguities

        for document in (
            "Patient Name: Harold Nkemdirim\nMRN: RGT/44219/B\nDOB: 12/03/1948\n",
            "Patient Name:\nHarold Nkemdirim\nMRN:\nRGT/44219/B\n",
            "Patient Name: John Michael Smith    MRN: RGT/44219/B\n",
            "Patient Name: Mary Parkinson\n",
            "Consultant: Dr Alan Rankin\n",
        ):
            assert not find_ambiguities(document), (
                f"an ordinary banner was refused: {document!r}"
            )


class TestTheLineStructureSurvives:
    """Nothing may be joined, split or deleted. Structural, not incidental."""

    @pytest.mark.parametrize("case", _CASES, ids=_ids(_CASES))
    def test_the_line_count_is_unchanged(self, case: Case) -> None:
        from mao.core.deident.text import split_lines

        before, _ = split_lines(case.text)
        after, _ = split_lines(scrub_pii(case.text))
        assert len(before) == len(after), (
            f"{case.id}: line count {len(before)} -> {len(after)}\n"
            f"--- in  ---\n{case.text!r}"
        )


class TestASecondPassIsSafe:
    r"""What idempotence can and cannot be, stated honestly.

    ## Why the unconditional property is not achievable

    A second pass can only recognise the first pass's work from something in the
    text, and the caller controls the text. Two mechanisms were tried:

        trust a MARKER    `_already_redacted` read `[NAME]` beside a label and
                          declared the field done. A caller who writes
                          `Patient Name: [NAME] Harold Nkemdirim` then suppresses
                          the redaction of the real name — all ten emittable
                          placeholders worked, for the two field types with no
                          second line of defence.

        trust a POSITION  `_value_is_placeholder` marked a label satisfied when
                          its own candidate CELL was placeholder-only. A caller
                          who INSERTS a placeholder-only line shifts the column
                          alignment and gets the same leak — three of four
                          shapes, a regression introduced by the fix for the
                          first mechanism.

    Both are forgeable because both ask caller-controlled text to certify what
    THIS process did. The property that is actually wanted — "this span was
    produced by this run" — is a fact about the run, so it lives in the run:
    `mao/trust/inputs/boundary.py` transforms each channel exactly once and
    carries the typed result, and `tests/agents/test_query_decomposer.py`
    asserts the second call site is gone.

    ## What still holds, and is asserted here

    1. A second pass never LEAKS. Whatever else it does, it cannot re-introduce
       an identifier the first pass removed. This is the safety direction and it
       is unconditional.
    2. A second pass is a NO-OP for every document the first pass left
       unambiguous — which is the overwhelming majority, and is exactly the
       population where a marker was never needed.
    3. Where the first pass left an ambiguity, a second pass may over-redact.
       That is the chat-path policy applied twice, it is visible in the answer,
       and it is recorded as a residual rather than papered over with a
       mechanism a caller can forge.
    """

    @pytest.mark.parametrize("case", _CASES, ids=_ids(_CASES))
    def test_a_second_pass_never_leaks(self, case: Case) -> None:
        once = scrub_pii(case.text)
        twice = visible(scrub_pii(once))
        survivors = [
            identifier
            for identifier in case.must_not_survive
            if visible(identifier) in twice
        ]
        assert not survivors, (
            f"{case.id}: a second pass re-exposed {survivors}\n"
            f"--- x1 ---\n{once!r}\n--- x2 ---\n{scrub_pii(once)!r}"
        )

    @pytest.mark.parametrize("case", _CASES, ids=_ids(_CASES))
    def test_a_second_pass_is_a_no_op_where_the_first_left_no_ambiguity(
        self, case: Case
    ) -> None:
        from mao.core.deident.ambiguity import find_ambiguities

        once = scrub_pii(case.text)
        if find_ambiguities(once):
            return  # covered by the residual above, and by the refusal policy
        assert scrub_pii(once) == once, (
            f"{case.id}: the first pass left nothing ambiguous, so the second "
            "pass had nothing to decide and should have changed nothing\n"
            f"--- x1 ---\n{once!r}\n--- x2 ---\n{scrub_pii(once)!r}"
        )

    def test_the_no_op_population_is_most_of_the_corpus(self) -> None:
        """The exclusion above must not be where the whole corpus goes."""
        from mao.core.deident.ambiguity import find_ambiguities

        stable = [c for c in _CASES if not find_ambiguities(scrub_pii(c.text))]
        share = len(stable) / len(_CASES)
        assert share > 0.6, (
            f"only {share:.0%} of the corpus reaches an unambiguous state after "
            "one pass, so the idempotence assertion above covers too little to "
            "mean anything"
        )


class TestTheAxesInteract:
    """Each finding was found one axis at a time; real inputs vary several."""

    _COMBINED: ClassVar[list[Case]] = combined_documents()

    @pytest.mark.parametrize("case", _COMBINED, ids=_ids(_COMBINED))
    def test_a_document_deviating_on_several_axes_still_holds_both_invariants(
        self, case: Case
    ) -> None:
        from mao.core.deident.ambiguity import find_ambiguities

        scrubbed = scrub_pii(case.text)
        seen = visible(scrubbed)
        survivors = [
            identifier
            for identifier in case.must_not_survive
            if visible(identifier) in seen
        ]
        # The destroy direction has the same two permitted outcomes here as
        # everywhere else: the content survives, or the document is refused.
        # See `TestNoClinicalContentIsDestroyed` for why.
        lost = [
            phrase
            for phrase in case.must_survive
            if phrase not in scrubbed and not find_ambiguities(case.text)
        ]
        assert not survivors and not lost, (
            f"{case.id}: leaked={survivors} destroyed={lost}\n"
            f"--- in  ---\n{case.text!r}\n--- out ---\n{seen!r}"
        )


class TestTheGuardItselfCanFail:
    """A probe that cannot fail proves nothing. Mutation controls.

    Every assertion above is run against a deliberately broken scrubber. If the
    corpus still passes, the corpus is not measuring anything.
    """

    def test_the_corpus_detects_a_disabled_scrubber(self) -> None:
        leaked = [
            case.id
            for case in _CASES
            if any(visible(i) in visible(case.text) for i in case.must_not_survive)
        ]
        assert len(leaked) > 400, (
            f"only {len(leaked)} hold-out documents would fail against an "
            "identity scrubber — the corpus is not asserting enough"
        )

    def test_the_visible_projection_sees_through_every_invisible_character(self) -> None:
        for name, character in INVISIBLE_CHARACTERS.items():
            spelled = f"Nkem{character}dirim"
            assert visible(spelled) == "Nkemdirim", (
                f"{name} survives the test's own visible() projection, so an "
                "assertion using it would pass a partial redaction"
            )

    def test_each_builder_contributes_documents(self) -> None:
        for builder in (
            name_boundary_documents,
            middle_name_documents,
            decorated_value_documents,
            forged_placeholder_documents,
            invisible_character_documents,
            clinical_preservation_documents,
            combined_documents,
        ):
            assert builder(), f"{builder.__name__} generated nothing"
