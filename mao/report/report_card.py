from __future__ import annotations
import io
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class MedicationEntry:
    name: str
    dose: str
    evidence: str
    notes: str = ""


@dataclass
class SourceEntry:
    tool: str
    snippet: str
    query: str


@dataclass
class ReportCard:
    stage: str
    stage_interpretation: str
    clinical_significance: str
    medications: list[MedicationEntry]
    literature_evidence: list[str]
    recommended_next_steps: list[str]
    confidence_score: float
    sources: list[SourceEntry]
    disclaimer: str = (
        "This output is for clinical decision support only. "
        "A licensed physician must review all findings before acting."
    )
    uncertainty_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_pdf(self) -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story = []

        def h(text: str) -> Paragraph:
            return Paragraph(f"<b>{text}</b>", styles["Heading2"])

        def p(text: str) -> Paragraph:
            return Paragraph(text, styles["Normal"])

        story += [h("MAO Clinical Report Card"), Spacer(1, 0.3*cm)]

        if self.uncertainty_flag:
            story += [
                Paragraph(
                    "<font color='red'><b>LOW CONFIDENCE - MRI model confidence below threshold. Treat findings with caution.</b></font>",
                    styles["Normal"]
                ),
                Spacer(1, 0.2*cm),
            ]

        story += [h("Stage"), p(f"{self.stage} - {self.stage_interpretation}"), Spacer(1, 0.3*cm)]
        story += [h("Clinical Significance"), p(self.clinical_significance), Spacer(1, 0.3*cm)]

        if self.medications:
            story.append(h("Medications"))
            data = [["Drug", "Dose", "Evidence", "Notes"]] + [
                [m.name, m.dose, m.evidence, m.notes] for m in self.medications
            ]
            table = Table(data, colWidths=[4*cm, 3*cm, 5*cm, 4*cm])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story += [table, Spacer(1, 0.3*cm)]

        if self.literature_evidence:
            story.append(h("Literature Evidence"))
            for ref in self.literature_evidence:
                story.append(p(f"- {ref}"))
            story.append(Spacer(1, 0.3*cm))

        if self.recommended_next_steps:
            story.append(h("Recommended Next Steps"))
            for step in self.recommended_next_steps:
                story.append(p(f"- {step}"))
            story.append(Spacer(1, 0.3*cm))

        story += [h("Confidence Score"), p(f"{self.confidence_score:.0%}"), Spacer(1, 0.3*cm)]
        story += [Paragraph(f"<i>{self.disclaimer}</i>", styles["Normal"])]

        doc.build(story)
        return buf.getvalue()


def build_report_card(**kwargs: Any) -> ReportCard:
    meds = [MedicationEntry(**m) if isinstance(m, dict) else m for m in kwargs.pop("medications", [])]
    srcs = [SourceEntry(**s) if isinstance(s, dict) else s for s in kwargs.pop("sources", [])]
    return ReportCard(medications=meds, sources=srcs, **kwargs)
