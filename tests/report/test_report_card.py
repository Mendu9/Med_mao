import pytest
from mao.report.report_card import ReportCard, MedicationEntry, SourceEntry, build_report_card

def _sample_card() -> ReportCard:
    return ReportCard(
        stage="EMCI",
        stage_interpretation="Early mild cognitive impairment — early intervention recommended.",
        clinical_significance="MMSE score 24/30 indicates mild impairment.",
        medications=[
            MedicationEntry(name="Donepezil", dose="5 mg/day", evidence="RCT (n=473)", notes="First-line cholinesterase inhibitor")
        ],
        literature_evidence=["Smith et al. 2023 — amyloid PET correlation with MMSE"],
        recommended_next_steps=["Repeat neuropsychological testing in 6 months", "Consider amyloid PET scan"],
        confidence_score=0.87,
        sources=[SourceEntry(tool="retriever", snippet="amyloid plaques are a hallmark...", query="amyloid AD")],
        disclaimer="This output is for clinical decision support only. A licensed physician must review.",
        uncertainty_flag=False,
    )

def test_report_card_fields():
    card = _sample_card()
    assert card.stage == "EMCI"
    assert card.confidence_score == 0.87
    assert len(card.medications) == 1
    assert card.medications[0].name == "Donepezil"

def test_to_dict_serialisable():
    import json
    card = _sample_card()
    d = card.to_dict()
    json.dumps(d)  # must not raise

def test_to_pdf_returns_bytes():
    card = _sample_card()
    pdf_bytes = card.to_pdf()
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:4] == b"%PDF"

def test_build_report_card_minimal():
    card = build_report_card(
        stage="CN",
        stage_interpretation="Cognitively normal.",
        clinical_significance="No significant impairment detected.",
        medications=[],
        literature_evidence=[],
        recommended_next_steps=["Annual follow-up"],
        confidence_score=0.95,
        sources=[],
        uncertainty_flag=False,
    )
    assert card.stage == "CN"

def test_uncertainty_flag_propagates():
    card = build_report_card(
        stage="AD",
        stage_interpretation="Alzheimer's disease.",
        clinical_significance="Significant impairment.",
        medications=[],
        literature_evidence=[],
        recommended_next_steps=[],
        confidence_score=0.45,
        sources=[],
        uncertainty_flag=True,
    )
    assert card.uncertainty_flag is True
