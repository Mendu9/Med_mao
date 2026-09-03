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
from collections.abc import Callable

#: Any of the three terminators, captured so it can be put back.
_TERMINATOR = re.compile(r"\r\n|\r|\n")

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


def strip_leading_bom(text: str) -> tuple[str, str]:
    """Detach a leading byte-order mark so it can be restored afterwards.

    Removing it outright would be simpler, but the invariant is that the output
    differs from the input ONLY where an identifier was removed. A BOM is not an
    identifier, so it goes back exactly where it was.
    """
    if text.startswith("﻿"):
        return "﻿", text[1:]
    return "", text
