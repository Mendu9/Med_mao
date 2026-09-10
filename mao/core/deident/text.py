r"""Line handling that survives the transport the document arrived over.

Two defects, both found by independent reviewers at the first Wave 11 SHA, were
really one missing abstraction:

  - CRLF input DISABLED THE ENTIRE LABELLED PATH. `layout` split on `"\n"`, so
    every line kept a trailing `\r`; `_is_label_only` stripped a separator set
    that did not contain `\r`, and every whole-line value match failed on the
    carriage return. `scrub_pii('Patient Name:\r\nMRN:\r\nHarold Nkemdirim\r\n')`
    came back completely unchanged, and the raw name then reached the
    application log and the database column named `pii_scrubbed_query`.
    pypdf only ever emits `\n`, so the generated-layout fuzzer could not
    produce this: `/chat` takes `ChatRequest.query` straight from an HTTP client
    with no newline normalisation, and a clinician pasting from Word sends CRLF.

  - THE FREE-TEXT RULES JOINED LINES. Four of them contain `\s`, which matches
    `\n`, and they ran over the whole document at once. A column of lab values
    (`138\n102\n2024`) collapsed into a single `[PHONE]`, deleting two lines of
    clinical data. `layout`'s "no line is ever joined" guarantee was real but
    LOCAL, and the module docstring claimed it for the whole scrubber.

So line scope lives here, once, and both passes are built on it. Splitting keeps
each terminator with its line and reassembly puts it back, so the transform is
byte-exact everywhere it does not redact — CRLF stays CRLF, a final line without
a terminator stays that way, and no rule can see across a line boundary at all.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

#: Every character Unicode treats as a line break, not just the three ASCII
#: ones. `str.splitlines()` breaks on all of these, and so does a PDF viewer —
#: but `re.split(r"\r\n|\r|\n")` did not, so a document using U+2028 LINE
#: SEPARATOR, U+2029, VT, FF or U+0085 NEL arrived as ONE line, no label line
#: was recognisable, and the whole labelled path was disabled exactly as CRLF
#: had disabled it.
_TERMINATOR = re.compile("\r\n|[\n\r\v\f  ]")

#: Invisible characters that are not whitespace to Python but are not content
#: either: BOM, zero-width space/non-joiner/joiner, word joiner, soft hyphen.
#: `str.strip()` leaves every one of them, so a leading BOM was enough to stop a
#: label line being recognised as a label line.
INVISIBLE = "﻿​‌‍⁠­"


def split_lines(text: str) -> tuple[list[str], list[str]]:
    """Split into content lines and the terminator that followed each.

    `len(contents) == len(terminators)`; the last terminator is `""` when the
    text does not end with one. Rejoining is exact.
    """
    contents: list[str] = []
    terminators: list[str] = []
    position = 0
    for match in _TERMINATOR.finditer(text):
        contents.append(text[position : match.start()])
        terminators.append(match.group())
        position = match.end()
    contents.append(text[position:])
    terminators.append("")
    return contents, terminators


def join_lines(contents: list[str], terminators: list[str]) -> str:
    return "".join(
        content + terminator
        for content, terminator in zip(contents, terminators, strict=True)
    )


def map_lines(text: str, transform: Callable[[str], str]) -> str:
    """Apply `transform` to each line's content, keeping every terminator.

    This is what makes "no line is ever joined, split or deleted" a property of
    the whole scrubber rather than of one pass: a transform applied here cannot
    see a line boundary, so it cannot cross one.
    """
    contents, terminators = split_lines(text)
    return join_lines([transform(content) for content in contents], terminators)


#: Every character that occupies NO VISIBLE ADVANCE WIDTH.
#:
#: This has now been enumerated three times and each enumeration was defeated by
#: a character the enumerator had not seen:
#:
#:   a six-element string   -> nine further characters split every grammar
#:                             (U+200E/200F, U+202A/202E, U+2061, U+2066,
#:                             U+061C, U+FE0F, U+180E), 54 of 90 combinations
#:   the `Cf` category      -> eight further characters, five of them named
#:                             verbatim in the review that suggested `Cf`:
#:                             U+034F (Mn), U+17B4/U+17B5 (Mn), the Hangul
#:                             fillers U+115F/U+1160/U+3164/U+FFA0 (Lo), the
#:                             Braille blank U+2800 (So), and the C0 controls
#:                             (Cc), which are legal in a JSON string body and
#:                             reach `ChatRequest.query` intact
#:
#: `Cf` was the right instinct applied to a proper subset. The property actually
#: wanted is "carries no visible content", and it spans five general categories
#: — so the class is defined by that property and the categories are how it is
#: computed, rather than the categories being the definition.
#:
#: A denylist of categories will keep losing. This is still, strictly, a
#: denylist — but it is one whose membership test is the property itself, so a
#: character added to Unicode tomorrow in any of these categories is handled
#: without anyone noticing it. The complementary control is in `values`: an
#: identifier token admits only the characters its own grammar names, so a
#: character that gets past this set still cannot sit INSIDE a claimed value.
def _carries_no_visible_content(character: str) -> bool:
    category = unicodedata.category(character)
    if category in ("Cf", "Cc"):
        # Format and control characters. `Cc` includes the line terminators,
        # which are structure rather than content and must survive: removing
        # them would join two lines, which is the deletion this module's whole
        # line-scope design exists to make impossible.
        return character not in "\t\n\r\v\f"
    if category in ("Mn", "Me"):
        # Non-spacing and enclosing marks: zero advance width by definition.
        # This covers the variation selectors (U+FE00.., U+E0100..), U+034F
        # COMBINING GRAPHEME JOINER and the Khmer inherent vowels — and also
        # ordinary combining accents, which is correct: `José` and `José`
        # must fold to the same token, and `_fold` already strips them for
        # lexicon lookup.
        return True
    if category == "Cs":
        # Lone surrogates. Not text, and no renderer draws them.
        return True
    # The remainder are individually named because their categories are
    # overwhelmingly visible: `Lo` letters and `So` symbols. Every one of these
    # renders as blank and every one was measured splitting an identifier.
    return character in _BLANK_BY_EXCEPTION


#: Blank characters whose general category is otherwise a visible one.
_BLANK_BY_EXCEPTION = frozenset(
    "ᅟᅠㅤﾠ"  # Hangul fillers (Lo) — the classic filter bypass
    "⠀"                    # Braille pattern blank (So)
    "᠎"                    # Mongolian vowel separator, reclassified out of Cf
    "­"                    # Soft hyphen, if ever reclassified
)

_FORMAT_CHARACTERS = frozenset(
    chr(code) for code in range(0x110000) if _carries_no_visible_content(chr(code))
)

_INVISIBLE_RE = re.compile(
    "[" + "".join(re.escape(character) for character in sorted(_FORMAT_CHARACTERS)) + "]"
)


def normalise(text: str) -> str:
    """NFKC, and remove invisible formatting characters everywhere.

    Folding invisibles inside NAME tokens only was not enough: PDF hyphenation
    emits U+00AD, and pypdf hands it straight through, so `RGT/44219<U+00AD>/B`
    split a record number in two and the whole identifier leaked — MRN, NHS
    number, phone, date and postcode alike. No attacker is needed to produce it.

    Removing them DOES change the text, and the invariant is that output differs
    from input only where an identifier was removed. The invariant is therefore
    stated against the NORMALISED input, and the generated-layout tests compare
    against `normalise(extracted)` for exactly that reason. A soft hyphen or a
    zero-width joiner is a rendering hint, not something a clinician wrote, and
    every de-identification tool worth the name folds it before matching.
    """
    return _INVISIBLE_RE.sub("", unicodedata.normalize("NFKC", text))


def strip_leading_bom(text: str) -> tuple[str, str]:
    """Detach a leading byte-order mark so it can be restored afterwards.

    Removing it outright would be simpler, but the invariant is that the output
    differs from the input ONLY where an identifier was removed. A BOM is not an
    identifier, so it goes back exactly where it was.
    """
    if text.startswith("﻿"):
        return "﻿", text[1:]
    return "", text
