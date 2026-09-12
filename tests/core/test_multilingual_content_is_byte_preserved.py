r"""O2 — meaning-bearing Unicode outside a redacted span is byte-preserved.

## The property

For text in any script, `scrub_with_report(text).text` must be byte-identical to
`text` everywhere outside a span the scrubber actually redacted. Not
"semantically equivalent", not "normalised the same way" — the same bytes.

## Why this is a separate oracle from O1

O1 asks whether the output agrees with the transformation's own record. It would
stay green on a scrubber that recorded "I replaced this whole document" and then
did. O2 asks the different question: is the content that a clinician wrote, in
the script they wrote it in, still there?

At `ff34722` it was not. `normalise()`'s folded output was the shipped string,
so:

  - 608 of 615 precomposed Latin letters were altered in shipped output;
  - every Greek and Cyrillic diacritic was dropped;
  - `año` shipped as `ano` and `café au lait` as `cafe au lait`;
  - Devanagari, Thai, Telugu and Sinhala vowel signs were deleted, so
    `बुखार` ("fever") shipped as
    `बखार` and `มี` ("has") lost its vowel.

None of that is detectable by an oracle that compares normalised forms, which is
why this one compares bytes.

## How this oracle is independent

Three ways, and each matters:

  the reference is the INPUT
      The expected answer is not computed by any function in `mao/`. It is the
      literal source string the test wrote down. There is no "what should the
      scrubber output be" step that could inherit the scrubber's own idea of
      canonical form.

  the comparison is `==` on `str`
      Python string equality is codepoint equality. Nothing is normalised on
      either side. `NFC(a) == NFC(b)` would pass on exactly the corpus this
      file exists to fail.

  the sweep is DERIVED, not listed
      `_precomposed_latin_letters` walks the Unicode database and takes every
      Latin letter whose NFD is longer than one codepoint. No human chose which
      letters to test, which is the failure mode `source.py`'s own docstring
      records for the enumeration it replaced.

## Non-vacuity

`_pre_fix_normalise` reconstructs the OLD behaviour from the description above —
NFD, drop `Mn`/`Me` marks of combining class zero, fold marks sitting on ASCII
bases and marks with a precomposed form — and `TestTheCorpusCanSeeTheOldDefect`
asserts the corpus DOES flag it. Without that, "0 of N altered" is a fact about
the instrument and not about the code.
"""
from __future__ import annotations

import unicodedata

import pytest

from mao.core.pii_scrubber import scrub_with_report

# --------------------------------------------------------------------------
# The corpus. One clinically plausible sentence per script, carrying the marks
# and vowel signs that the pre-fix pipeline destroyed.
# --------------------------------------------------------------------------

SENTENCES: dict[str, str] = {
    # --- Latin with diacritics ---
    "french": "Le patient présente une céphalée sévère et une fièvre élevée.",
    "spanish": "El paciente tiene años de dolor y una pequeña lesión cutánea.",
    "german": "Der Patient hat starke Kopfschmerzen und Schwäche in den Füßen.",
    "vietnamese": "Bệnh nhân bị đau đầu dữ dội và sốt cao.",
    "turkish": "Hasta şiddetli baş ağrısı ve yüksek ateş bildirdi.",
    "polish": "Pacjent zgłasza silną gorączkę i złe samopoczucie.",
    "czech": "Pacient má silné bolesti hlavy a vysokou horečku.",
    "portuguese": "O paciente apresenta cefaleia intensa, febre e náuseas persistentes.",
    "icelandic": "Sjúklingurinn er með höfuðverk og háan hita í dag.",
    "hungarian": "A beteg erős fejfájásról és magas lázról számol be.",
    # --- other alphabets ---
    "greek": "Ο ασθενής αναφέρει έντονη κεφαλαλγία και υψηλό πυρετό.",
    "cyrillic": "Пациент жалуется на сильную головную боль и высокую температуру.",
    # --- Indic and South-East Asian: the vowel signs the blanket Mn strip ate ---
    "devanagari": "रोगी को तेज़ बुखार और सिरदर्द है।",
    "bengali": "রোগীর তীব্র জ্বর এবং মাথাব্যথা আছে।",
    "tamil": "நெரியாளருக்கு கடுமையான காய்ச்சல் உள்ளது.",
    "telugu": "రోగికి తీవ్రమైన జ్వరం మరియు తలనొప్పి ఉంది.",
    "kannada": "ರೋಗಿಯಲ್ಲಿ ಅಧಿಕ ಜ್ವರ ಮತ್ತು ತಲೆನೋವು ಇದೆ.",
    "sinhala": "රෝගියාට තද උණුසුමක් සහ හිසරදයක් ඇත.",
    "thai": "ผู้ป่วยมีไข้สูงและปวดศีรษะมาก",
    "lao": "ຜູ້ປ່ວຍມີໄຂ້ສູງແລະເຈັບຫົວ.",
    "khmer": "អ្នកជំងឺមានគ្រុនច្រើន។",
    "myanmar": "လူနာသည် အပြင်းအလွန် ဖြစ်နေသည်။",
    # --- abjads with points / harakat ---
    "hebrew_pointed": "הַחוֹלֶה מִתׅלוֹנֵן עַל כְאֵב רֹאשׁ חָזָק.",
    "arabic_harakat": "يَشْكُو الْمَرِيضُ مِنْ صُدَاعٍ شَدِيدٍ.",
    # --- CJK and Hangul ---
    "hangul": "환자는 심한 두통과 고열을 호소합니다.",
    "japanese": "患者は強い頭痛と高熱を訴えています。",
    "chinese": "患者主诉剧烈头痛和高热。",
}

#: Named individually because these are the exact strings the pre-fix pipeline
#: was measured destroying. A regression here is a regression to `ff34722`.
NAMED_REGRESSIONS = {
    "ano": "El paciente refiere dolor desde hace años.",
    "cafe_au_lait": "Multiple café au lait spots noted on the trunk.",
    "bukhar": "बुखार",
    "thai_mi": "มี",
}


def _precomposed_latin_letters() -> tuple[str, ...]:
    """Every Latin letter whose canonical decomposition carries a mark.

    DERIVED from the Unicode database rather than listed. Nobody chose these,
    so nobody can have chosen them to pass.
    """
    found: list[str] = []
    for code in range(0x2500):
        character = chr(code)
        if not unicodedata.category(character).startswith("L"):
            continue
        if len(unicodedata.normalize("NFD", character)) <= 1:
            continue
        if "LATIN" not in unicodedata.name(character, ""):
            continue
        found.append(character)
    return tuple(found)


PRECOMPOSED_LATIN = _precomposed_latin_letters()


def _pre_fix_normalise(text: str) -> str:
    """The pre-`41cfabd` shipped transform, reconstructed from its description.

    NFD, then drop every `Mn`/`Me` mark that either has combining class zero,
    or sits on an ASCII alphanumeric base, or composes back into its base —
    then NFKC. This is the behaviour whose output used to BE the scrubbed text.

    Written here, in the test, rather than imported: importing the real
    `_fold_decorations` would make the control a statement about today's code
    instead of about the defect it is supposed to reproduce.
    """
    decomposed = unicodedata.normalize("NFD", text)
    out: list[str] = []
    for character in decomposed:
        if unicodedata.category(character) in ("Mn", "Me"):
            if unicodedata.combining(character) == 0:
                continue
            if out and out[-1].isascii() and out[-1].isalnum():
                continue
            if out and len(unicodedata.normalize("NFC", out[-1] + character)) == 1:
                continue
        out.append(character)
    return unicodedata.normalize("NFKC", "".join(out))


# --------------------------------------------------------------------------
# Oracle helpers. Nothing below calls into `mao.core.deident`.
# --------------------------------------------------------------------------


def _outside_removed_spans(source: str, result) -> list[str]:
    """The pieces of `source` the scrubber did NOT claim to have removed."""
    pieces: list[str] = []
    cursor = 0
    for removal in sorted(result.removals, key=lambda item: item.start):
        pieces.append(source[cursor : removal.start])
        cursor = removal.end
    pieces.append(source[cursor:])
    return [piece for piece in pieces if piece]


class TestTextWithNoIdentifierIsUntouched:
    """The simple half: nothing to redact, so nothing may change."""

    @pytest.mark.parametrize("script", sorted(SENTENCES))
    def test_the_sentence_is_byte_identical(self, script: str) -> None:
        source = SENTENCES[script]
        produced = scrub_with_report(source).text
        assert produced == source, (
            f"{script}: the scrubber altered text containing no identifier.\n"
            f"  in  : {source!r}\n  out : {produced!r}"
        )

    @pytest.mark.parametrize("name", sorted(NAMED_REGRESSIONS))
    def test_the_named_ff34722_regressions_do_not_recur(self, name: str) -> None:
        source = NAMED_REGRESSIONS[name]
        assert scrub_with_report(source).text == source

    def test_no_script_is_altered(self) -> None:
        """The headline count, stated as a count so the report can quote it."""
        altered = [
            script
            for script, source in SENTENCES.items()
            if scrub_with_report(source).text != source
        ]
        assert altered == [], f"{len(altered)} of {len(SENTENCES)} scripts altered: {altered}"


class TestContentOutsideARedactedSpanIsByteIdentical:
    """The harder half: a real redaction happens in the same document."""

    @pytest.mark.parametrize("script", sorted(SENTENCES))
    def test_the_non_latin_sentence_survives_beside_an_identifier(
        self, script: str
    ) -> None:
        sentence = SENTENCES[script]
        source = (
            "Patient Name: Harold Nkemdirim\n"
            f"{sentence}\n"
            "MRN: A1234567\n"
        )
        result = scrub_with_report(source)

        assert result.removals, (
            f"{script}: no identifier was removed, so this case is not "
            "measuring preservation ACROSS a redaction"
        )
        assert sentence in result.text, (
            f"{script}: the sentence did not survive byte-identically beside a "
            f"redaction.\n  wanted : {sentence!r}\n  got    : {result.text!r}"
        )
        for piece in _outside_removed_spans(source, result):
            assert piece in result.text, (
                f"{script}: a stretch of source outside every recorded removal "
                f"span is missing from the output: {piece!r}"
            )

    @pytest.mark.parametrize("script", sorted(SENTENCES))
    def test_the_identifier_really_went(self, script: str) -> None:
        """Non-vacuity for the case above: preservation is only interesting if
        the scrubber was doing its job in the same document."""
        source = (
            "Patient Name: Harold Nkemdirim\n"
            f"{SENTENCES[script]}\n"
            "MRN: A1234567\n"
        )
        produced = scrub_with_report(source).text
        assert "Harold Nkemdirim" not in produced
        assert "A1234567" not in produced
        assert "[NAME]" in produced


class TestEveryPrecomposedLatinLetterSurvives:
    """The breadth sweep. Derived from the Unicode database, not chosen."""

    CARRIER = "Finding noted {letter} in the clinical record."

    def test_the_sweep_is_wide_enough_to_mean_something(self) -> None:
        assert len(PRECOMPOSED_LATIN) >= 400, (
            f"only {len(PRECOMPOSED_LATIN)} precomposed Latin letters found — "
            "the derivation is broken and the sweep proves nothing"
        )

    def test_zero_precomposed_latin_letters_are_altered(self) -> None:
        altered = [
            letter
            for letter in PRECOMPOSED_LATIN
            if scrub_with_report(self.CARRIER.format(letter=letter)).text
            != self.CARRIER.format(letter=letter)
        ]
        assert altered == [], (
            f"{len(altered)} of {len(PRECOMPOSED_LATIN)} precomposed Latin "
            f"letters were altered in shipped output, e.g. {altered[:10]!r}"
        )

    def test_they_survive_beside_a_redaction_too(self) -> None:
        """A subset, in a document that actually redacts something.

        Every hundredth letter rather than all of them: the point is that the
        redaction path and the preservation path are the same path, and that
        does not need 500 repetitions to establish.
        """
        altered = []
        for letter in PRECOMPOSED_LATIN[::37]:
            sentence = f"Lesion described as {letter} on examination."
            source = f"Patient Name: Harold Nkemdirim\n{sentence}\nMRN: A1234567\n"
            if sentence not in scrub_with_report(source).text:
                altered.append(letter)
        assert altered == [], f"altered beside a redaction: {altered!r}"


class TestTheCorpusCanSeeTheOldDefect:
    """NON-VACUITY CONTROL.

    Everything above reports zero. That number is a fact about the scrubber only
    if the same corpus, fed the OLD behaviour, reports a large number. If the
    control were also zero, the corpus would be the thing that is broken.
    """

    def test_the_old_behaviour_alters_the_precomposed_latin_sweep(self) -> None:
        carrier = TestEveryPrecomposedLatinLetterSurvives.CARRIER
        altered = [
            letter
            for letter in PRECOMPOSED_LATIN
            if _pre_fix_normalise(carrier.format(letter=letter))
            != carrier.format(letter=letter)
        ]
        assert len(altered) == len(PRECOMPOSED_LATIN), (
            "the reconstructed pre-fix transform did NOT destroy every "
            f"precomposed Latin letter ({len(altered)} of "
            f"{len(PRECOMPOSED_LATIN)}) — the control is not reproducing the "
            "defect it claims to"
        )

    def test_the_old_behaviour_alters_most_of_the_script_corpus(self) -> None:
        altered = [
            script
            for script, source in SENTENCES.items()
            if _pre_fix_normalise(source) != source
        ]
        assert len(altered) >= 15, (
            f"the corpus only flags {len(altered)} of {len(SENTENCES)} scripts "
            "under the old behaviour — it is too weak to have proved anything "
            f"by reporting zero under the new one. Flagged: {altered}"
        )

    @pytest.mark.parametrize("name", sorted(NAMED_REGRESSIONS))
    def test_the_named_regressions_are_reproduced_by_the_control(
        self, name: str
    ) -> None:
        """The four strings the remediation note names, each one flagged."""
        source = NAMED_REGRESSIONS[name]
        assert _pre_fix_normalise(source) != source, (
            f"{name}: the control does not reproduce this named regression, so "
            f"the corresponding green assertion is unproven ({source!r})"
        )

    def test_the_control_reproduces_the_exact_named_outputs(self) -> None:
        """Not merely 'different' — the specific strings the note recorded."""
        assert _pre_fix_normalise("año") == "ano"
        assert _pre_fix_normalise("café au lait") == "cafe au lait"
        assert _pre_fix_normalise("बुखार") == "बखार"
        assert _pre_fix_normalise("มี") == "ม"
