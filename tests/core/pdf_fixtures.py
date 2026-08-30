"""Real clinic letters, rendered to PDF and extracted with pypdf.

Wave 9 / B2. Three consecutive waves were defeated by a hand-written fixture
whose shape the implementer chose: `tests/api/test_phi_never_reaches_the_provider.py`
built the one layout — label and value on the same line — that the scrubber
happened to handle, and the gate passed while five other real layouts leaked.

So these fixtures are not strings. Each one is drawn with reportlab and read
back with pypdf, which is the exact pair the production path uses
(`clinical_agent._extract_pdf_text`). The shape under test is whatever pypdf
actually produces, not whatever the test author imagined it produces. Both
libraries are hard requirements in `requirements.txt`, so this needs no skip.

A two-column letterhead is the important case: pypdf emits `label \n value`,
which is why `_VALUE` matched the empty string and the scrubber printed
`Patient Name: [NAME]` directly above the untouched real name.
"""
from __future__ import annotations

import io

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

# Identifiers that must never survive scrubbing. Synthetic, but shaped like the
# real thing: a double-barrelled surname, a slashed MRN, a UK mobile.
NAME = "Jonathan Aldred-Whitmore"
KIN = "Margaret Aldred-Whitmore"
MRN = "RGT/44219/B"
DOB = "12/03/1948"
TEL = "07700 900123"

IDENTIFIERS = (NAME, KIN, MRN, DOB, TEL)

# Clinical content that must survive scrubbing. The bradycardia line is the one
# the B1 probe proved was silently deleted before the model ever saw it.
CLINICAL_LINES = (
    "Impression: Progressive amnestic syndrome consistent with early",
    "Alzheimer's disease. MMSE 22/30. Background of sinus bradycardia",
    "at 48 bpm, currently untreated.",
)


def _extract(draw, font: str = "Helvetica") -> str:
    """Render with reportlab, read back with pypdf — the production pair."""
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    pdf.setFont(font, 10)
    draw(pdf)
    pdf.save()
    buf.seek(0)
    return "\n".join(page.extract_text() or "" for page in PdfReader(buf).pages)


def _clinical(pdf, y: int) -> None:
    for line in CLINICAL_LINES:
        pdf.drawString(60, y, line)
        y -= 14


def two_column() -> str:
    """A real clinic letterhead: labels left, values right.

    pypdf reads the columns as separate runs, so every value lands on the line
    *after* its label. This is the layout that leaked 5 of 6 in Wave 8.
    """
    def draw(pdf):
        y = 800
        for label, value in (
            ("Patient Name:", NAME),
            ("MRN:", MRN),
            ("Date of Birth:", DOB),
            ("Tel:", TEL),
        ):
            pdf.drawString(60, y, label)
            pdf.drawString(220, y, value)
            y -= 18
        _clinical(pdf, y - 20)

    return _extract(draw)


def single_column() -> str:
    """Label and value on one line — the only shape the old fixture built."""
    def draw(pdf):
        y = 800
        for line in (
            f"Patient Name: {NAME}",
            f"MRN: {MRN}",
            f"Date of Birth: {DOB}",
            f"Tel: {TEL}",
        ):
            pdf.drawString(60, y, line)
            y -= 16
        _clinical(pdf, y - 16)

    return _extract(draw)


def labels_then_values() -> str:
    """Wide two-column: every label extracts before any value.

    The degenerate form of `two_column`. Pairing here needs the run of labels
    matched against the run of values that follows it.
    """
    def draw(pdf):
        y = 800
        for label in ("Patient Name:", "MRN:", "Date of Birth:"):
            pdf.drawString(60, y, label)
            y -= 18
        y = 800
        for value in (NAME, MRN, DOB):
            pdf.drawString(320, y, value)
            y -= 18
        _clinical(pdf, 700)

    return _extract(draw)


def fullwidth_colon() -> str:
    """U+FF1A instead of U+003A — routine in scanned and CJK-locale documents.

    pypdf preserves the fullwidth codepoint, so a scrubber keyed on ASCII ':'
    matches nothing at all. NFKC folds it; `input_guardrails` normalises but
    `clinical_agent` did not, which is B3.
    """
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

    def draw(pdf):
        y = 800
        for line in (
            f"Patient Name：{NAME}",
            f"MRN：{MRN}",
            f"Date of Birth：{DOB}",
        ):
            pdf.drawString(60, y, line)
            y -= 18
        _clinical(pdf, y - 16)

    return _extract(draw, font="HeiseiMin-W3")


def prose_only() -> str:
    """No field labels anywhere — the name appears in running narrative."""
    def draw(pdf):
        y = 800
        for line in (
            "Dear Dr Fairbanks,",
            f"I reviewed {NAME} in clinic today. He is a 77-year-old",
            "gentleman with a two-year history of progressive memory decline.",
            f"His wife {KIN} attended with him.",
            "Alzheimer's disease. MMSE 22/30. Background of sinus bradycardia.",
        ):
            pdf.drawString(60, y, line)
            y -= 16

    return _extract(draw)


#: Every layout, by name. Parametrise over this so a new layout is covered
#: everywhere at once.
LAYOUTS = {
    "two_column": two_column,
    "single_column": single_column,
    "labels_then_values": labels_then_values,
    "fullwidth_colon": fullwidth_colon,
    "prose_only": prose_only,
}
