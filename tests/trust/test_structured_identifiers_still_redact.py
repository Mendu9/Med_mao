r"""O6 — ordinary identifiers still redact, and ADV16-1 stays closed.

Three properties, and the third is the one that keeps the first two honest.

## 1. Ordinary letterheads still work

`Patient Name:`, `MRN:`, `NHS Number:`, `DOB:`, `Postcode:`, `Telephone:`,
`Email:` and `NI Number:` across several layouts — own line, inline banner,
pipe-separated, two-column with wide gaps, value-before-label. The value is
replaced by the right placeholder, and the clinical content on the same document
is untouched. Both halves matter: a scrubber that redacts the letterhead by
deleting the page satisfies the first and fails the second.

## 2. ADV16-1 stays closed

A combining mark inserted at EVERY alphanumeric position inside a labelled
specimen, for each of ten marks, across six identifier types. The defect was
that `normalise()` composed before it stripped, so a mark with a precomposed
form was folded into its base before the strip could see it, and an accented
postcode reached a third-party model with no grammar matching it.

The `Cc` half is the same attack with `U+0001` and `U+0007`, which are legal in
a JSON string body and reach `ChatRequest.query` intact. They are written as
`\uXXXX` escapes, and `test_the_control_specimens_are_not_ascii` FAILS if an
editor round-trip ever turns them back into ASCII — which is how three earlier
reproductions of this quietly stopped reproducing anything.

## 3. The probe can see a leak

`TestTheProbeCanSeeALeak` rebinds a deliberately broken normaliser — NFKC
before the strip, the pre-fix ordering — and asserts the SAME probe reports
leaks. A zero-leak result from an instrument that cannot report a non-zero one
is a fact about the instrument.

## How this oracle is independent

  the leak test is a normalised substring search written here
      `_readable` strips marks and format characters with `unicodedata` and asks
      whether the undecorated identifier is a substring. It does not call
      `text.normalise`, `gateway._visible`, or any detection code in `mao/`.
      That matters because the point of ADV16-1 was that a control and its
      backstop shared a detection step and neither could catch what the other
      missed.

  the probe positions are enumerated, not chosen
      Every alphanumeric index of every specimen, crossed with every mark. No
      position was selected by hand, so none can have been selected to pass.
"""
from __future__ import annotations

import unicodedata

import pytest

import mao.core.pii_scrubber as pii_scrubber
from mao.core.deident.source import MatchView
from mao.core.pii_scrubber import scrub_with_report

# --------------------------------------------------------------------------
# Part 1 — ordinary letterheads
# --------------------------------------------------------------------------

CLINICAL_LINE = "Complete heart block with a permanent pacemaker in situ."
SECOND_CLINICAL_LINE = "Donepezil 10mg od was commenced after review."

FIELDS = {
    "Patient Name": ("Harold Nkemdirim", "[NAME]"),
    "MRN": ("RGT/44219/B", "[MRN]"),
    "NHS Number": ("943 476 5919", "[NHS]"),
    "DOB": ("12/03/1948", "[DOB]"),
    "Postcode": ("SW1A 1AA", "[POSTCODE]"),
    "Telephone": ("020 7946 0958", "[PHONE]"),
    "NI Number": ("QQ123456C", "[NI_NUMBER]"),
}


def _layouts(label: str, value: str) -> dict[str, str]:
    """One field rendered several ways, each with clinical content beside it."""
    return {
        "own_line": f"{label}: {value}\n{CLINICAL_LINE}\n{SECOND_CLINICAL_LINE}\n",
        "inline_banner": (
            f"{label}: {value}   Ward: Elm\n{CLINICAL_LINE}\n"
        ),
        "pipe_separated": (
            f"{label}: {value} | Ward: Elm | Consultant: on call\n{CLINICAL_LINE}\n"
        ),
        "two_column_wide_gap": (
            f"{label}:{' ' * 20}{value}\n{CLINICAL_LINE}\n"
        ),
        "label_line_then_value_line": (
            f"{label}:\n{value}\n{CLINICAL_LINE}\n"
        ),
        "value_before_label": f"{value}  {label}\n{CLINICAL_LINE}\n",
        "value_then_dashed_label": f"{value} - {label}\n{CLINICAL_LINE}\n",
    }


class TestOrdinaryLetterheadsStillRedact:
    @pytest.mark.parametrize("label", sorted(FIELDS))
    @pytest.mark.parametrize(
        "layout",
        [
            "own_line",
            "inline_banner",
            "pipe_separated",
            "two_column_wide_gap",
            "label_line_then_value_line",
            "value_before_label",
            "value_then_dashed_label",
        ],
    )
    def test_the_value_is_replaced_by_its_placeholder(
        self, label: str, layout: str
    ) -> None:
        value, placeholder = FIELDS[label]
        document = _layouts(label, value)[layout]
        produced = scrub_with_report(document).text

        assert value not in produced, (
            f"{label} / {layout}: the raw value survived.\n"
            f"  in : {document!r}\n  out: {produced!r}"
        )
        assert placeholder in produced, (
            f"{label} / {layout}: no {placeholder} in the output, so the value "
            f"was removed without being typed.\n  out: {produced!r}"
        )

    @pytest.mark.parametrize("label", sorted(FIELDS))
    @pytest.mark.parametrize(
        "layout",
        [
            "own_line",
            "inline_banner",
            "pipe_separated",
            "two_column_wide_gap",
            "label_line_then_value_line",
            "value_before_label",
            "value_then_dashed_label",
        ],
    )
    def test_the_clinical_content_is_untouched(
        self, label: str, layout: str
    ) -> None:
        """The other half of the invariant, and the one that keeps failing."""
        value, _ = FIELDS[label]
        document = _layouts(label, value)[layout]
        produced = scrub_with_report(document).text
        assert CLINICAL_LINE in produced, (
            f"{label} / {layout}: clinical content was altered or deleted.\n"
            f"  out: {produced!r}"
        )

    def test_a_full_letterhead_redacts_every_field_at_once(self) -> None:
        document = (
            "\n".join(f"{label}: {value}" for label, (value, _) in FIELDS.items())
            + f"\n{CLINICAL_LINE}\n{SECOND_CLINICAL_LINE}\n"
        )
        produced = scrub_with_report(document).text
        leaked = [value for value, _ in FIELDS.values() if value in produced]
        assert leaked == [], f"raw identifiers survived a full letterhead: {leaked}"
        assert CLINICAL_LINE in produced
        assert SECOND_CLINICAL_LINE in produced


class TestABracketedTrailingLabelDefeatsTheValueBeforeLabelPath:
    r"""REPORTED PRODUCT DEFECT — a parenthesised trailing label leaks.

    `mao/core/pii_scrubber.py` names value-before-label among the layouts the
    field grammar covers, and it does: `Harold Nkemdirim  Patient Name`,
    `Harold Nkemdirim - Patient Name` and `Harold Nkemdirim | Patient Name` are
    all redacted. So is a bracketed label in the ordinary order,
    `(Patient Name): Harold Nkemdirim`.

    What is NOT redacted is a value followed by a BRACKETED label:

        'Harold Nkemdirim (Patient Name)'  ->  unchanged
        'A1234567 (MRN)'                   ->  unchanged
        'Harold Nkemdirim [Patient Name]'  ->  unchanged

    A full patient name and a record number survive the scrubber. This is the
    shape a form export or a table-to-text conversion produces, and the egress
    backstop cannot catch it either: `RequestProtection.leaked_in` can only
    assert identifiers the scrubber RECORDED removing, and it recorded none.

    ## What bounds it

    An identifier with a STANDALONE shape rule is still caught, because the
    free-text pass does not need the label at all: `943 476 5919 (NHS Number)`
    redacts via the modulus-11 rule, and a postcode, telephone number, DOB or
    NI number likewise. The exposure is exactly the types that only the labelled
    path can find — a person's name, and a local record number whose format no
    issuing body fixes. Those are also the two most identifying fields on the
    letterhead, so the bound is narrow comfort.

    `test_a_shape_detectable_identifier_is_caught_anyway` asserts that bound, so
    the report above is a measured claim rather than an impression.

    Reported, not patched — the fix belongs in `mao/core/deident/fields.py` or
    `layout.py`, which this worktree may not touch.
    """

    LEAKY = (
        "Harold Nkemdirim (Patient Name)",
        "A1234567 (MRN)",
        "Harold Nkemdirim [Patient Name]",
    )

    #: Caught by the shape pass regardless of how the label is written.
    SHAPE_DETECTABLE = (
        ("943 476 5919 (NHS Number)", "943 476 5919"),
        ("SW1A 1AA (Postcode)", "SW1A 1AA"),
        ("020 7946 0958 (Telephone)", "020 7946 0958"),
        ("QQ123456C (NI Number)", "QQ123456C"),
    )
    SOUND = (
        "(Patient Name): Harold Nkemdirim",
        "[Patient Name] Harold Nkemdirim",
        "Harold Nkemdirim - Patient Name",
        "Harold Nkemdirim | Patient Name",
    )

    @pytest.mark.parametrize("document", SOUND)
    def test_an_unbracketed_or_leading_bracketed_label_redacts(
        self, document: str
    ) -> None:
        produced = scrub_with_report(document).text
        assert "Harold Nkemdirim" not in produced, (
            f"{document!r} -> {produced!r}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT DEFECT: a value followed by a bracketed label is not "
            "recognised, so the raw name or record number survives."
        ),
    )
    @pytest.mark.parametrize("document", LEAKY)
    def test_a_bracketed_trailing_label_also_redacts(self, document: str) -> None:
        produced = scrub_with_report(document).text
        leaked = [
            value
            for value in ("Harold Nkemdirim", "A1234567", "943 476 5919")
            if value in produced
        ]
        assert leaked == [], f"{document!r} -> {produced!r} leaked {leaked}"

    @pytest.mark.parametrize(("document", "value"), SHAPE_DETECTABLE)
    def test_a_shape_detectable_identifier_is_caught_anyway(
        self, document: str, value: str
    ) -> None:
        """The bound on the defect, asserted rather than assumed."""
        produced = scrub_with_report(document).text
        assert value not in produced, (
            f"{document!r} -> {produced!r}: the shape pass did not catch this "
            "either, so the defect is WIDER than reported"
        )

    def test_the_defect_is_still_present_as_described(self) -> None:
        assert scrub_with_report(self.LEAKY[0]).text == self.LEAKY[0], (
            "the bracketed trailing label is now handled — remove the xfail"
        )


class TestALabelledEmailIsRedactedWhole:
    r"""REPORTED PRODUCT DEFECT — a partial email redaction.

    On the plainest letterhead layout the labelled path claims a SHORTER span
    than the email occupies, and the shape path — which gets it right — never
    sees the remainder, because it runs on text in which the labelled span has
    already been staged out:

        'Email: harold@nhs.uk'       ->  'Email: [EMAIL]nhs.uk'
        'Email: h.n@example.nhs.uk'  ->  'Email: [EMAIL]nhs.uk'
        'Contact harold@nhs.uk now'  ->  'Contact [EMAIL] now'   (correct)

    `Email Address:` additionally mistypes it as `[ADDRESS]`.

    The recorded `Removal.value` is `harold@`, so the egress backstop is told to
    watch for the wrong string. The mailbox itself is removed, so this is not a
    disclosure of the individual — the residue is an organisation domain — but
    it is the "placeholder covering PART of an identifier" shape the design
    explicitly forbids, and the labelled path being WORSE than the unlabelled
    one on the same input is the inversion worth reporting.

    Reported, not patched.
    """

    @pytest.mark.parametrize(
        "document",
        [
            "Email: harold@nhs.uk",
            "Email: h.nkemdirim@example.nhs.uk",
            "E-mail: harold@nhs.uk",
        ],
    )
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT DEFECT: the labelled EMAIL span stops short and leaves the "
            "domain tail standing."
        ),
    )
    def test_no_fragment_of_the_address_survives(self, document: str) -> None:
        produced = scrub_with_report(document).text
        assert "nhs.uk" not in produced, f"{document!r} -> {produced!r}"
        assert "@" not in produced

    def test_the_unlabelled_path_gets_it_right(self) -> None:
        """The comparison that makes the defect a defect and not a policy."""
        produced = scrub_with_report("Contact harold@nhs.uk for the result.").text
        assert produced == "Contact [EMAIL] for the result."

    def test_the_defect_is_still_present_as_described(self) -> None:
        assert scrub_with_report("Email: harold@nhs.uk").text == "Email: [EMAIL]nhs.uk"


class TestANelTerminatedLetterheadRedacts:
    r"""REPORTED PRODUCT DEFECT — CRITICAL. U+0085 disables the labelled path.

    `text._TERMINATOR` counts U+0085 NEL as a line terminator.
    `text._carries_no_visible_content` does not exempt it: NEL is general
    category `Cc`, and the exemption is the literal set `"\t\n\r\v\f"`. So
    `build_match_view` strips every NEL, the document becomes ONE line in the
    match view, and the labelled path mis-associates from the first field on.

        'Patient Name: Harold Nkemdirim<NEL>MRN: RGT/44219/B<NEL>NHS Number: ...'
          ->  'Patient Name: [NAME]: RGT/44219/B<NEL>NHS Number: 943 476 5919...'

    The `[NAME]` span swallowed the terminator and the next line's `MRN` label,
    and every labelled identifier after the first survives RAW.

    ## Both walls are blind to it

    The only thing recorded against the request's protection is
    `('NAME', 'Harold Nkemdirim\x85MRN')`. `RequestProtection.leaked_in` on the
    protected text therefore returns `[]` — the backstop can only assert
    identifiers the scrubber RECORDED removing, exactly as `report.py` says
    ("a removal record can only contain what the scrubber found"). Measured
    end to end through `POST /chat` with a provider bound at
    `gateway.set_provider`: the raw MRN and the raw NHS number reached the
    provider seam.

    U+2028 and U+2029 are unaffected because they are `Zl`/`Zp`. NEL is the one
    terminator the module's two halves disagree about — and `text.py`'s own
    docstring names U+0085 among the terminators whose loss "disabled the whole
    labelled path exactly as CRLF had disabled it".

    Reported, not patched — the fix is one character class in
    `mao/core/deident/text.py`.
    """

    NEL = ""
    DOCUMENT = (
        "Patient Name: Harold Nkemdirim"
        "MRN: RGT/44219/B"
        "NHS Number: 943 476 5919"
        "Telephone: 020 7946 0958"
        "Complete heart block."
    )
    IDENTIFIERS = ("Harold Nkemdirim", "RGT/44219/B", "943 476 5919",
                   "020 7946 0958")

    def test_the_specimen_really_contains_a_nel(self) -> None:
        """Meta-test, same reasoning as the `Cc` specimens below."""
        assert self.NEL in self.DOCUMENT
        assert unicodedata.category(self.NEL) == "Cc"

    def test_the_same_document_with_a_newline_redacts_everything(self) -> None:
        """The control. Only the terminator differs."""
        produced = scrub_with_report(self.DOCUMENT.replace(self.NEL, "\n")).text
        leaked = [value for value in self.IDENTIFIERS if value in produced]
        assert leaked == [], f"the LF form leaks too: {leaked} in {produced!r}"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT DEFECT (critical): U+0085 NEL is stripped by the match "
            "view, collapsing the document to one line, and every labelled "
            "identifier after the first survives in full."
        ),
    )
    def test_no_identifier_survives_a_nel_terminated_letterhead(self) -> None:
        produced = scrub_with_report(self.DOCUMENT).text
        leaked = [value for value in self.IDENTIFIERS if value in produced]
        assert leaked == [], (
            f"raw identifiers survived: {leaked}\n  out: {produced!r}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRODUCT DEFECT (critical): the run-scoped egress assertion cannot "
            "see the leak, because the scrubber never recorded the identifiers."
        ),
    )
    def test_the_egress_backstop_sees_the_leak(self) -> None:
        from mao.trust.classes import InputChannel
        from mao.trust.egress.gateway import RequestProtection, protected_request
        from mao.trust.inputs.boundary import protect_channel

        protection = RequestProtection(trace_id="o6-nel")
        with protected_request(protection):
            protected = protect_channel(
                self.DOCUMENT, InputChannel.REPORT, refuse_ambiguity=False
            )
            assert protection.leaked_in(protected.text), (
                "the backstop reports no leak on a payload carrying a raw MRN, "
                f"a raw NHS number and a raw telephone number: {protected.text!r}"
            )

    def test_the_defect_is_still_present_as_described(self) -> None:
        """Pin it, so the two xfails above are checkable claims."""
        produced = scrub_with_report(self.DOCUMENT).text
        leaked = [value for value in self.IDENTIFIERS if value in produced]
        assert "RGT/44219/B" in leaked, (
            "the MRN no longer leaks on the NEL path — the defect may be "
            "fixed, in which case remove the xfails above"
        )
        assert "943 476 5919" in leaked
        assert produced.count(self.NEL) < self.DOCUMENT.count(self.NEL), (
            "NEL terminators are no longer deleted"
        )


# --------------------------------------------------------------------------
# Part 2 — ADV16-1
# --------------------------------------------------------------------------

#: Ten combining marks. Four have a precomposed form with an ASCII vowel and six
#: do not with most bases, which is the distinction that defeated the pre-fix
#: ordering: `Q` has no precomposed acute, so a precomposed-form test alone was
#: never sufficient.
COMBINING_MARKS = (
    "́",  # COMBINING ACUTE ACCENT
    "̀",  # COMBINING GRAVE ACCENT
    "̂",  # COMBINING CIRCUMFLEX ACCENT
    "̃",  # COMBINING TILDE
    "̈",  # COMBINING DIAERESIS
    "̧",  # COMBINING CEDILLA
    "̱",  # COMBINING MACRON BELOW
    "̄",  # COMBINING MACRON
    "̆",  # COMBINING BREVE
    "̊",  # COMBINING RING ABOVE
)

#: The `Cc` half. Written as escapes on purpose — see the meta-test below.
CONTROL_CHARACTERS = (
    "",  # START OF HEADING
    "",  # BELL
)

#: `(document, identifier)`. The identifier is what must not be readable.
SPECIMENS = (
    ("Postcode: SW1A 1AA", "SW1A 1AA"),
    ("MRN: A1234567", "A1234567"),
    ("NHS Number: 943 476 5919", "943 476 5919"),
    ("Telephone: 020 7946 0958", "020 7946 0958"),
    ("NI Number: QQ123456C", "QQ123456C"),
    ("Email: h.nkemdirim@example.com", "h.nkemdirim@example.com"),
)


def _readable(text: str) -> str:
    """`text` with everything that can hide inside an identifier taken out.

    Written here rather than imported. `mao.core.deident.text.normalise` is the
    function under test and `gateway._visible` is its backstop; using either
    would make this probe share a detection step with the thing it grades, which
    is the ADV16-1 mechanism itself.
    """
    decomposed = unicodedata.normalize("NFD", text)
    kept = [
        character
        for character in decomposed
        if unicodedata.category(character) not in ("Mn", "Me", "Cf", "Cc")
    ]
    return unicodedata.normalize("NFKC", "".join(kept))


def _decorated_probes(decorations: tuple[str, ...]):
    """Every `(document, identifier)` with one decoration at one position."""
    for document, identifier in SPECIMENS:
        base = document.index(identifier)
        for offset, character in enumerate(identifier):
            if not character.isalnum():
                continue
            for decoration in decorations:
                cut = base + offset + 1
                yield document[:cut] + decoration + document[cut:], identifier


def _leaks(probes) -> list[tuple[str, str]]:
    """Probes on which the undecorated identifier is still readable."""
    found: list[tuple[str, str]] = []
    for document, identifier in probes:
        produced = scrub_with_report(document).text
        haystack = _readable(produced).replace(" ", "")
        if identifier.replace(" ", "") in haystack:
            found.append((document, produced))
    return found


MARK_PROBES = tuple(_decorated_probes(COMBINING_MARKS))
CONTROL_PROBES = tuple(_decorated_probes(CONTROL_CHARACTERS))


class TestADV16OneStaysClosed:
    def test_the_probe_set_is_large_enough(self) -> None:
        assert len(MARK_PROBES) >= 400, f"only {len(MARK_PROBES)} mark probes"
        assert len(CONTROL_PROBES) >= 80, f"only {len(CONTROL_PROBES)} Cc probes"

    def test_no_combining_mark_lets_an_identifier_through(self) -> None:
        leaks = _leaks(MARK_PROBES)
        assert leaks == [], (
            f"{len(leaks)} of {len(MARK_PROBES)} decorated identifiers leaked, "
            f"e.g. {leaks[:5]!r}"
        )

    def test_no_control_character_lets_an_identifier_through(self) -> None:
        leaks = _leaks(CONTROL_PROBES)
        assert leaks == [], (
            f"{len(leaks)} of {len(CONTROL_PROBES)} control-decorated "
            f"identifiers leaked, e.g. {leaks[:5]!r}"
        )

    def test_the_undecorated_specimens_redact(self) -> None:
        """Non-vacuity: if the plain forms did not redact, the decorated result
        would say nothing about decoration."""
        for document, identifier in SPECIMENS:
            produced = scrub_with_report(document).text
            assert identifier not in produced, (
                f"the UNDECORATED form leaks, so this file is not measuring "
                f"ADV16-1 at all: {document!r} -> {produced!r}"
            )

    def test_a_placeholder_is_emitted_for_every_decorated_probe(self) -> None:
        """Removal without typing is a different failure from a leak.

        Sampled rather than exhaustive: this is a shape check, and the leak
        sweep above is the exhaustive one.
        """
        untyped = [
            document
            for document, _ in MARK_PROBES[::17]
            if "[" not in scrub_with_report(document).text
        ]
        assert untyped == [], (
            f"{len(untyped)} decorated probes produced no placeholder at all, "
            f"e.g. {untyped[:3]!r}"
        )


class TestTheControlSpecimensAreNotAscii:
    """META-TEST. Three earlier reproductions of this stopped reproducing.

    `U+0001` and `U+0007` are invisible in every editor. A copy-paste, a
    sanitiser or an encoding round-trip can replace them with nothing, and the
    file keeps passing while testing a plain ASCII string. This fails loudly if
    that ever happens.
    """

    def test_the_control_characters_are_control_characters(self) -> None:
        for character in CONTROL_CHARACTERS:
            assert not character.isascii() or unicodedata.category(character) == "Cc", (
                f"{character!r} is not a control character any more"
            )
            assert unicodedata.category(character) == "Cc", (
                f"{character!r} has category {unicodedata.category(character)}, "
                "not Cc — this specimen has been round-tripped away"
            )

    def test_no_control_probe_is_pure_ascii_text(self) -> None:
        """The probe documents must actually contain a control character."""
        plain = [
            document
            for document, _ in CONTROL_PROBES
            if all(
                unicodedata.category(character) != "Cc"
                for character in document
            )
        ]
        assert plain == [], (
            f"{len(plain)} control-character probes contain no Cc character at "
            f"all, so they are testing undecorated input: {plain[:3]!r}"
        )

    def test_the_mark_probes_really_carry_a_mark(self) -> None:
        plain = [
            document
            for document, _ in MARK_PROBES[::29]
            if all(
                unicodedata.category(character) not in ("Mn", "Me")
                for character in document
            )
        ]
        assert plain == [], f"mark probes with no mark: {plain[:3]!r}"

    def test_the_specimens_differ_from_their_probes(self) -> None:
        """A decoration that changed nothing would make every probe a duplicate
        of the undecorated specimen, and the sweep would grade nothing."""
        originals = {document for document, _ in SPECIMENS}
        assert not (originals & {document for document, _ in MARK_PROBES})
        assert not (originals & {document for document, _ in CONTROL_PROBES})


# --------------------------------------------------------------------------
# Part 3 — the control that proves the zero is about the code
# --------------------------------------------------------------------------


def _broken_match_view(source: str) -> MatchView:
    """The PRE-FIX ordering: compose FIRST, then strip.

    `normalise()`'s defect was exactly this order. NFKC folds a combining mark
    into its base before the strip can see it, so the strip could only ever fire
    for a (base, mark) pair Unicode has no precomposed form for — and a decorated
    ASCII identifier therefore reached the grammars still decorated.

    Written here, not imported, so the control describes the DEFECT rather than
    today's code. The map is the identity, which is all the leak probe needs.
    """
    from mao.core.deident.text import _INVISIBLE_RE

    composed = unicodedata.normalize("NFKC", source)
    text = _INVISIBLE_RE.sub("", composed)
    if len(text) != len(source):
        # Keep the map trivially valid; the probe only reads whether a grammar
        # matched, and a length change here would only make the control weaker.
        text = source
    return MatchView(
        source=source,
        text=text,
        _starts=tuple(range(len(source))),
        _ends=tuple(range(1, len(source) + 1)),
    )


class TestTheProbeCanSeeALeak:
    """NON-VACUITY CONTROL for the whole of Part 2.

    Zero leaks out of several hundred probes is a claim about the scrubber only
    if the same probe, run against a scrubber with the pre-fix ordering,
    reports a non-zero number.
    """

    @pytest.fixture()
    def broken_normaliser(self, monkeypatch):
        monkeypatch.setattr(
            pii_scrubber, "build_match_view", _broken_match_view
        )

    def test_the_broken_ordering_leaks(self, broken_normaliser) -> None:
        leaks = _leaks(MARK_PROBES[:120])
        assert leaks, (
            "the probe reports ZERO leaks even against a normaliser with the "
            "pre-fix ordering, so the zero it reports against the real one is "
            "a fact about the probe and not about the code"
        )

    def test_the_broken_ordering_leaks_a_lot(self, broken_normaliser) -> None:
        """Not a single accident — the defect class, reproduced at scale."""
        sample = MARK_PROBES[:120]
        leaks = _leaks(sample)
        assert len(leaks) >= len(sample) // 4, (
            f"only {len(leaks)} of {len(sample)} probes leak under the broken "
            "ordering, which is too few to establish that the probe is "
            "sensitive to this defect class"
        )

    def test_the_named_adv16_1_specimens_leak_under_the_broken_ordering(
        self, broken_normaliser
    ) -> None:
        """The three the finding named, each one exhibited."""
        acute = "́"
        named = (
            (f"Postcode: SW1{acute}A 1AA", "SW1A 1AA"),
            (f"MRN: A{acute}1234567", "A1234567"),
            (f"NI Number: Q{acute}Q123456C", "QQ123456C"),
        )
        leaks = _leaks(named)
        assert len(leaks) == 3, (
            f"only {len(leaks)} of the 3 named ADV16-1 specimens leak under the "
            f"broken ordering: {leaks!r}"
        )

    def test_the_real_normaliser_is_restored_afterwards(self) -> None:
        """Guards against the control poisoning the rest of the session."""
        acute = "́"
        assert _leaks(((f"Postcode: SW1{acute}A 1AA", "SW1A 1AA"),)) == []
        assert pii_scrubber.build_match_view.__name__ == "build_match_view"
