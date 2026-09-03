"""Generative clinical-document layouts: reportlab -> pypdf, ground truth attached.

Wave 11. A fixture list the implementer chooses has defeated this project four
times (Waves 5, 6, 7 and 9). Every one of those lists was written *after* the
implementer knew which shapes the scrubber handled, so it encoded the same blind
spot twice and the gate passed while real layouts leaked.

So the layouts here are not chosen. They are the cartesian product of the axes a
clinical document actually varies along, rendered with reportlab and read back
with pypdf — the exact pair `clinical_agent._extract_pdf_text` uses — so the
shape under test is whatever pypdf really produces:

    separator      colon, fullwidth colon, pipe, dash, none, wide column gap
    order          label before value, value before label
    columns        inline, two-column, all-labels-then-all-values, table row
    orphans        0, 1 or 2 labels with no value anywhere in the document
    clinical       narrative before / after / interleaved with the header
    field set      identity, contact, national-identifier

Each generated document carries its own ground truth, and the ground truth is
*measured against the extracted text*, never assumed: an identifier pypdf broke
across lines is dropped from the expectation rather than asserted about, so a
failure always means the scrubber failed and never that the renderer did.

Two invariants are asserted on every document (see test_pii_scrubber_layouts.py):

  (a) no raw identifier survives to the scrubbed string;
  (b) no clinical line is deleted or altered, and no placeholder is emitted
      unless a correctly typed identifier was actually removed there.

Wave 9 tested (a) thoroughly and (b) only where no orphan label preceded the
clinical line, which is exactly the hole the Wave 10 CRITICAL lived in.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

# --- Vocabulary -------------------------------------------------------------
#
# Several values per kind, so a rule cannot be fitted to one string. All
# synthetic, all shaped like the real thing: double-barrelled and non-Anglophone
# surnames, a slashed MRN, a UK mobile, a spaced NHS number.

#: kind -> the placeholder types that may legitimately stand for a value of that
#: kind. A postcode inside an address may surface as either.
ALLOWED_PLACEHOLDERS: dict[str, frozenset[str]] = {
    "NAME": frozenset({"NAME"}),
    "MRN": frozenset({"MRN", "ID_NUMBER", "NHS_ID", "ACCOUNT"}),
    "NHS": frozenset({"NHS", "NHS_ID", "ID_NUMBER"}),
    "DOB": frozenset({"DOB"}),
    "PHONE": frozenset({"PHONE", "NHS_ID", "ID_NUMBER"}),
    "ADDRESS": frozenset({"ADDRESS", "POSTCODE"}),
    "EMAIL": frozenset({"EMAIL"}),
    "POSTCODE": frozenset({"POSTCODE"}),
    "NI_NUMBER": frozenset({"NI_NUMBER"}),
}

#: kind -> (label, value) pairs. The label vocabulary varies too: a scrubber
#: keyed on "Patient Name" alone must not pass by accident.
VOCABULARY: dict[str, tuple[tuple[str, str], ...]] = {
    "NAME": (
        ("Patient Name", "Harold Nkemdirim"),
        ("Name", "Jonathan Aldred-Whitmore"),
        ("Next of Kin", "Margaret Aldred-Whitmore"),
        ("Consultant", "Priya Venkataraman"),
    ),
    "MRN": (
        ("MRN", "RGT/44219/B"),
        ("Hospital Number", "LDS9931C"),
        ("Medical Record Number", "004512399"),
        ("Case No", "MC-2024-88231"),
    ),
    "NHS": (
        ("NHS Number", "943 476 5919"),
        ("NHS No", "4857773456"),
    ),
    "DOB": (
        ("Date of Birth", "12/03/1948"),
        ("DOB", "1948-03-12"),
        ("Born", "12 March 1948"),
    ),
    "PHONE": (
        ("Telephone", "07700 900123"),
        ("Mobile", "+44 7700 900456"),
        ("Contact Number", "0113 496 0231"),
    ),
    "ADDRESS": (
        ("Address", "22 Kingsway Avenue"),
        ("Home Address", "148 Beckett Road"),
    ),
    "EMAIL": (
        ("Email", "h.nkemdirim@example-nhs.uk"),
        ("E-mail", "margaret.aw@example.org"),
    ),
    "POSTCODE": (
        ("Postcode", "LS9 7TF"),
        ("Post Code", "SW1A 1AA"),
    ),
    "NI_NUMBER": (
        ("NI Number", "QQ123456C"),
        ("National Insurance", "AB654321D"),
    ),
}

#: The three header blocks a clinic letter is built from.
FIELD_SETS: dict[str, tuple[str, ...]] = {
    "identity": ("NAME", "MRN", "DOB"),
    "contact": ("PHONE", "ADDRESS", "EMAIL"),
    "national": ("NHS", "NI_NUMBER", "POSTCODE"),
}

# Clinical lines that must survive verbatim. Deliberately drawn from the shapes
# that defeat a "is this line value-like?" heuristic: bare medication lines,
# capitalised headings, an ALL-CAPS finding, a name-shaped drug, and the
# bradycardia line whose deletion is a contraindication for the drug the model
# is being asked about.
CLINICAL_CORPUS: tuple[str, ...] = (
    "Current medications",
    "Warfarin INR 2.4",
    "Atorvastatin 40mg",
    "Memantine 20mg daily",
    "Bradycardia 48 bpm untreated",
    "Donepezil 10 mg once daily",
    "MMSE 22/30",
    "BRADYCARDIA PRESENT",
    "Impression Progressive amnestic syndrome",
    "Scheltens scale 3",
    "Amyloid PET positive",
    "Alzheimer disease probable",
)

_SEPARATORS: dict[str, str] = {
    "colon": ": ",
    "fullwidth_colon": "：",
    "pipe": " | ",
    "dash": " - ",
    "none": " ",
    "wide_gap": "    ",
}
#: Separators reportlab's Helvetica cannot draw. Rendered with a CID font.
_NEEDS_CID = frozenset({"fullwidth_colon"})


@dataclass(frozen=True)
class Field:
    """One header field, and the ground truth about it."""

    kind: str
    label: str
    value: str


@dataclass(frozen=True)
class Layout:
    """One point in the cartesian product. `name` is the pytest test id."""

    separator: str
    order: str
    columns: str
    orphans: int
    clinical: str
    field_set: str

    @property
    def name(self) -> str:
        return (
            f"{self.field_set}-{self.columns}-{self.separator}-{self.order}"
            f"-orphan{self.orphans}-clinical_{self.clinical}"
        )


@dataclass(frozen=True)
class Document:
    """A rendered layout plus everything a property test needs to judge it.

    Every expectation is filtered against `extracted`: an identifier pypdf broke
    up is not asserted about, so a failure is always the scrubber's.
    """

    layout: Layout
    extracted: str
    #: Raw identifier strings verified present in `extracted`. None may survive.
    identifiers: tuple[str, ...]
    #: Clinical lines verified present in `extracted`. All must survive verbatim.
    clinical_lines: tuple[str, ...]
    #: Fields whose value is present in `extracted`, with the line it sits on.
    placed: tuple[tuple[Field, int], ...]
    #: Labels present with no value anywhere in the document. A placeholder next
    #: to one of these asserts a removal that never happened.
    orphan_labels: tuple[str, ...]
    #: (clinical line, index) for every clinical line occupying a whole line.
    #: Those lines carry no identifier, so scrubbing must leave them identical.
    clinical_at: tuple[tuple[str, int], ...]
    #: (orphan label, index) for every orphan label occupying a whole line.
    orphan_at: tuple[tuple[str, int], ...]

    @property
    def id(self) -> str:
        return self.layout.name


def _render(lines_at: list[tuple[float, float, str]], font: str) -> str:
    """Draw absolutely-positioned strings and read the page back with pypdf."""
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    pdf.setFont(font, 10)
    for x, y, text in lines_at:
        pdf.drawString(x, y, text)
    pdf.save()
    buf.seek(0)
    return "\n".join(page.extract_text() or "" for page in PdfReader(buf).pages)


def _fields_for(layout: Layout, index: int) -> tuple[Field, ...]:
    """Pick one (label, value) per kind, rotating so no single pair dominates."""
    fields = []
    for offset, kind in enumerate(FIELD_SETS[layout.field_set]):
        choices = VOCABULARY[kind]
        label, value = choices[(index + offset) % len(choices)]
        fields.append(Field(kind=kind, label=label, value=value))
    return tuple(fields)


def _orphan_labels_for(layout: Layout, index: int) -> tuple[str, ...]:
    """Labels drawn into the document with no value anywhere.

    A wide two-column extraction produces these naturally, and they are where
    the scrubber may not invent a placeholder.
    """
    if not layout.orphans:
        return ()
    pool = [
        VOCABULARY[kind][index % len(VOCABULARY[kind])][0]
        for kind in ("MRN", "NAME", "NHS", "PHONE")
    ]
    start = index % len(pool)
    rotated = pool[start:] + pool[:start]
    return tuple(rotated[: layout.orphans])


def _clinical_for(index: int) -> tuple[str, ...]:
    """Three clinical lines, rotating through the corpus."""
    n = len(CLINICAL_CORPUS)
    return tuple(CLINICAL_CORPUS[(index + k) % n] for k in range(3))


def _draw(
    layout: Layout,
    fields: tuple[Field, ...],
    orphans: tuple[str, ...],
    clinical: tuple[str, ...],
) -> list[tuple[float, float, str]]:
    """Position every string on the page according to the layout's axes."""
    sep = _SEPARATORS[layout.separator]
    placements: list[tuple[float, float, str]] = []

    def header_rows() -> list[tuple[str, str]]:
        """(left, right) for each header row, honouring label/value order."""
        rows: list[tuple[str, str]] = []
        for field in fields:
            if layout.order == "label_first":
                rows.append((field.label, field.value))
            else:
                rows.append((field.value, field.label))
        return rows

    y = 780.0
    header: list[tuple[float, float, str]] = []
    rows = header_rows()

    def cell(text: str, is_label: bool) -> str:
        """A column cell. Letterheads print the label with or without its colon;
        both are real, and only one of them was ever tested."""
        if not is_label or layout.separator == "none":
            return text
        return f"{text}:"

    if layout.columns == "inline":
        for left, right in rows:
            header.append((60.0, y, f"{left}{sep}{right}"))
            y -= 18
        for label in orphans:
            header.append((60.0, y, f"{label}{sep.rstrip()}".rstrip()))
            y -= 18
    elif layout.columns == "two_column":
        # Label left, value right on the same baseline. pypdf reads the columns
        # as separate runs, so each right-hand cell lands on its own line.
        for left, right in rows:
            label_left = layout.order == "label_first"
            header.append((60.0, y, cell(left, is_label=label_left)))
            header.append((260.0, y, cell(right, is_label=not label_left)))
            y -= 18
        for label in orphans:
            header.append((60.0, y, cell(label, is_label=True)))
            y -= 18
    elif layout.columns == "table_row":
        # A real table: EVERY label on one baseline, every value on the next.
        # pypdf emits each as a single line with wide gaps between the cells, so
        # neither line is a `label \n value` pair and neither is label-only.
        # A header row and a data row, each on ONE line with wide gaps between
        # the cells.
        #
        # Positioning the cells separately does NOT produce this: pypdf emits
        # text in content-stream order and gives each cell its own line, so a
        # reportlab-drawn table extracts as `labels_then_values`. This shape
        # arrives instead from PDFs whose text operators emit a whole row at
        # once, and from a clinician pasting a table into `/chat` — both of
        # which reach `scrub_pii` through exactly the same call.
        label_first = layout.order == "label_first"
        label_cells = [cell(left if label_first else right, is_label=True) for left, right in rows]
        # An orphan column: a header cell whose data cell is empty. Its type must
        # not be adopted by a neighbouring column's value.
        label_cells += [cell(label, is_label=True) for label in orphans]
        value_cells = [(right if label_first else left) for left, right in rows]
        # Spaces, never a tab: reportlab draws `\t` as a notdef glyph rather
        # than whitespace, so a tab here would assert on the renderer instead of
        # the scrubber. Tab-separated rows reach `scrub_pii` from pasted text and
        # are covered directly in test_pii_scrubber_value_bounds.py.
        gap = "   " if layout.separator == "none" else "    "
        first, second = (
            (label_cells, value_cells) if label_first else (value_cells, label_cells)
        )
        header.append((55.0, y, gap.join(first)))
        header.append((55.0, y - 18, gap.join(second)))
        y -= 40
    else:  # labels_then_values — the degenerate wide layout
        top = y
        for left, _ in rows:
            header.append((60.0, y, cell(left, is_label=layout.order == "label_first")))
            y -= 18
        for label in orphans:
            header.append((60.0, y, cell(label, is_label=True)))
            y -= 18
        bottom = y
        y = top
        for _, right in rows:
            header.append(
                (320.0, y, cell(right, is_label=layout.order != "label_first"))
            )
            y -= 18
        y = bottom

    clinical_rows: list[tuple[float, float, str]] = []
    for line in clinical:
        clinical_rows.append((60.0, y, line))
        y -= 16

    if layout.clinical == "after":
        placements = header + clinical_rows
    elif layout.clinical == "before":
        # Redraw with the narrative on top: shift the header down instead of
        # recomputing, which keeps the column geometry identical.
        shift = 16.0 * len(clinical)
        placements = [(x, yy - shift, t) for x, yy, t in header]
        placements += [(60.0, 780.0 - 16.0 * i, line) for i, line in enumerate(clinical)]
    else:  # interleaved — a clinical line between every header row
        placements = []
        yy = 780.0
        merged: list[str | tuple[float, float, str]] = []
        del merged
        head_iter = iter(header)
        clin_iter = iter(clinical_rows)
        # Interleave by baseline: take header rows in pairs, then one clinical
        # line, so narrative genuinely separates the header fields.
        pending_header = list(head_iter)
        pending_clinical = [t for _, _, t in clin_iter]
        while pending_header or pending_clinical:
            for _ in range(2):
                if pending_header:
                    x, _old, text = pending_header.pop(0)
                    placements.append((x, yy, text))
            yy -= 18
            if pending_clinical:
                placements.append((60.0, yy, pending_clinical.pop(0)))
                yy -= 16
    return placements


def _line_of(extracted_lines: list[str], value: str) -> int | None:
    for i, line in enumerate(extracted_lines):
        if value in line:
            return i
    return None


def _build(layout: Layout, index: int) -> Document | None:
    fields = _fields_for(layout, index)
    orphans = _orphan_labels_for(layout, index)
    clinical = _clinical_for(index)

    # An orphan label that collides with a real field's label is not an orphan.
    used = {field.label for field in fields}
    orphans = tuple(label for label in orphans if label not in used)

    font = "Helvetica"
    if layout.separator in _NEEDS_CID:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
        font = "HeiseiMin-W3"

    extracted = _render(_draw(layout, fields, orphans, clinical), font)
    lines = extracted.split("\n")

    placed: list[tuple[Field, int]] = []
    identifiers: list[str] = []
    for field in fields:
        at = _line_of(lines, field.value)
        if at is None:
            continue  # pypdf broke the value up; not this scrubber's problem
        placed.append((field, at))
        identifiers.append(field.value)

    surviving_clinical = tuple(line for line in clinical if line in extracted)
    surviving_orphans = tuple(label for label in orphans if label in extracted)

    # A whole-line occurrence is the strict case: the line carries nothing but
    # clinical text, or nothing but an unpaired label, so scrubbing must return
    # it byte-identical.
    clinical_at = tuple(
        (line, i)
        for line in surviving_clinical
        for i, text in enumerate(lines)
        if text.strip() == line
    )
    orphan_at = tuple(
        (label, i)
        for label in surviving_orphans
        for i, text in enumerate(lines)
        if text.strip() in (label, f"{label}:")
    )

    if not identifiers:
        return None
    return Document(
        layout=layout,
        extracted=extracted,
        identifiers=tuple(identifiers),
        clinical_lines=surviving_clinical,
        placed=tuple(placed),
        orphan_labels=surviving_orphans,
        clinical_at=clinical_at,
        orphan_at=orphan_at,
    )


def layouts() -> Iterator[Layout]:
    """Every point in the product. Combinations that render identically are
    deduplicated downstream by their extracted text."""
    for field_set in FIELD_SETS:
        for columns in ("inline", "two_column", "labels_then_values", "table_row"):
            for separator in _SEPARATORS:
                # A column layout has no in-line separator; the only variation
                # that survives extraction is whether the label cell keeps its
                # colon, so those layouts run under exactly `colon` and `none`.
                if columns != "inline" and separator not in ("colon", "none"):
                    continue
                for order in ("label_first", "value_first"):
                    for orphans in (0, 1, 2):
                        for clinical in ("after", "before", "interleaved"):
                            yield Layout(
                                separator=separator,
                                order=order,
                                columns=columns,
                                orphans=orphans,
                                clinical=clinical,
                                field_set=field_set,
                            )


def generate() -> list[Document]:
    """Render every layout once. Deterministic: no RNG anywhere."""
    documents: list[Document] = []
    seen: set[str] = set()
    for index, layout in enumerate(layouts()):
        document = _build(layout, index)
        if document is None or document.extracted in seen:
            continue
        seen.add(document.extracted)
        documents.append(document)
    return documents
