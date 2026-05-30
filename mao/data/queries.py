"""
mao/data/queries.py
-------------------
PMC search query definitions for bulk paper download.
Pure data module — no side effects on import.

Used by: mao/data/download_pmc.py

IMPORTANT: These queries are sent to db="pmc" via Entrez esearch.
- Do NOT include pmc[sb] — PubMed-only filter, returns 0 results in PMC db
- Do NOT include [dp] or [pt] — PubMed-only field tags
- Date filtering (2016:2026[pdat]) is appended in _search_pmc() automatically
"""
from __future__ import annotations
import re


SOURCE_GROUPS: dict[str, dict] = {
    "stroke": {
        "queries": [
            '"acute ischemic stroke" OR "acute ischaemic stroke" OR "ischemic stroke" OR "ischaemic stroke"',
            '("large vessel occlusion" OR LVO OR "middle cerebral artery occlusion" OR "internal carotid artery occlusion") AND ("ischemic stroke" OR "acute ischemic stroke")',
            '("CT perfusion" OR CTP OR "computed tomography perfusion") AND ("ischemic stroke" OR "acute ischemic stroke")',
            '("CT angiography" OR CTA OR "computed tomography angiography") AND ("ischemic stroke" OR "large vessel occlusion")',
            '("multimodal CT" OR "perfusion imaging") AND ("ischemic stroke" OR "acute ischemic stroke")',
            '("final infarct volume" OR "follow-up infarct" OR "final lesion") AND ("ischemic stroke" OR "acute ischemic stroke")',
            '("infarct volume prediction" OR "lesion prediction") AND ("CT perfusion" OR CTP OR CTA)',
            '("infarct growth" OR "lesion growth" OR "infarct progression") AND ("ischemic stroke" OR "acute ischemic stroke")',
            '("ischemic core" OR "infarct core" OR penumbra OR "ischemic penumbra" OR "tissue at risk") AND ("ischemic stroke" OR "acute ischemic stroke")',
            '("core penumbra mismatch" OR "perfusion mismatch" OR "target mismatch") AND ("ischemic stroke" OR "acute ischemic stroke")',
            '("cerebral blood flow" OR CBF OR "cerebral blood volume" OR CBV OR Tmax OR "time to maximum") AND ("CT perfusion" OR CTP)',
            '("perfusion threshold" OR "Tmax threshold" OR "CBF threshold" OR "ischemic core threshold") AND ("CT perfusion" OR CTP)',
            '("machine learning" OR "deep learning" OR "artificial intelligence") AND ("ischemic stroke" OR "acute ischemic stroke")',
            '("lesion segmentation" OR "stroke lesion segmentation" OR "infarct segmentation" OR "U-Net" OR nnU-Net) AND ("ischemic stroke" OR "brain infarct")',
            '"Ischemic Stroke Lesion Segmentation" OR ISLES',
            '("mechanical thrombectomy" OR thrombectomy OR EVT OR "endovascular therapy") AND ("ischemic stroke" OR "large vessel occlusion")',
            '(thrombolysis OR alteplase OR tenecteplase OR tPA OR "tissue plasminogen activator") AND ("ischemic stroke" OR "acute ischemic stroke")',
            '(reperfusion OR recanalization OR mTICI OR TICI) AND ("ischemic stroke" OR thrombectomy)',
            '("treatment selection" OR "patient selection" OR "imaging selection") AND ("ischemic stroke" OR "CT perfusion" OR thrombectomy)',
            '("modified Rankin Scale" OR mRS OR "functional outcome" OR prognosis OR mortality) AND ("ischemic stroke" OR thrombectomy OR thrombolysis)',
            '("hemorrhagic transformation" OR "symptomatic intracranial hemorrhage" OR sICH OR "cerebral edema") AND ("ischemic stroke" OR thrombectomy)',
            '("stroke rehabilitation" OR "post-stroke care" OR "secondary prevention") AND ("ischemic stroke" OR TIA)',
            '"ischemic stroke" AND "systematic review"',
            '"ischemic stroke" AND (guideline OR consensus OR recommendation)',
        ],
    },

    "alzheimers": {
        "queries": [
            '"Alzheimer disease" OR "Alzheimer\'s disease"',
            '("mild cognitive impairment" OR MCI OR "prodromal Alzheimer" OR "preclinical Alzheimer") AND ("Alzheimer disease" OR dementia)',
            '(dementia OR "cognitive impairment" OR "cognitive decline") AND ("Alzheimer disease" OR "Alzheimer\'s disease")',
            '("Alzheimer disease" OR MCI) AND (diagnosis OR prediction OR prognosis OR "disease progression")',
            '("mild cognitive impairment" OR MCI) AND ("conversion to Alzheimer" OR progression OR "cognitive decline")',
            '("early diagnosis" OR "preclinical Alzheimer" OR "diagnostic criteria") AND ("Alzheimer disease" OR MCI)',
            '("machine learning" OR "deep learning" OR "artificial intelligence") AND ("Alzheimer disease" OR MCI OR dementia)',
            '(classification OR prediction OR "risk model") AND ("Alzheimer disease" OR MCI) AND ("machine learning" OR "deep learning")',
            '("explainable AI" OR XAI OR interpretability) AND ("Alzheimer disease" OR MCI)',
            '(MRI OR "structural MRI" OR "brain MRI") AND ("Alzheimer disease" OR MCI)',
            '("hippocampal atrophy" OR "brain atrophy" OR "cortical thickness" OR "hippocampal volume") AND ("Alzheimer disease" OR MCI)',
            '(MRI OR "structural MRI") AND ("deep learning" OR "machine learning") AND ("Alzheimer disease" OR MCI)',
            '("amyloid PET" OR "tau PET" OR "FDG PET" OR "positron emission tomography") AND ("Alzheimer disease" OR MCI)',
            '("amyloid imaging" OR "amyloid PET") AND ("Alzheimer disease" OR MCI)',
            '("tau imaging" OR "tau PET" OR "tau pathology") AND ("Alzheimer disease" OR MCI)',
            '("amyloid beta" OR amyloid OR "amyloid positivity" OR Abeta) AND ("Alzheimer disease" OR MCI)',
            '(tau OR "phosphorylated tau" OR "p-tau" OR pTau OR "total tau") AND ("Alzheimer disease" OR MCI)',
            '"ATN framework" AND ("Alzheimer disease" OR "Alzheimer\'s disease")',
            '("CSF biomarker" OR "cerebrospinal fluid biomarker") AND ("Alzheimer disease" OR MCI)',
            '("plasma biomarker" OR "blood biomarker" OR "serum biomarker") AND ("Alzheimer disease" OR MCI OR dementia)',
            '("plasma p-tau" OR "p-tau181" OR "p-tau217" OR NfL OR GFAP) AND ("Alzheimer disease" OR dementia)',
            '(ADNI OR "Alzheimer\'s Disease Neuroimaging Initiative") AND (MRI OR PET OR CSF OR biomarker)',
            '("multimodal" OR "data fusion" OR "multimodal machine learning") AND ("Alzheimer disease" OR MCI) AND (MRI OR PET OR CSF OR biomarker)',
            '("MMSE" OR "Mini-Mental State Examination" OR "MoCA" OR "ADAS-Cog") AND ("Alzheimer disease" OR MCI OR dementia)',
            '(APOE OR APOE4 OR "genetic risk" OR "polygenic risk score") AND ("Alzheimer disease" OR "Alzheimer\'s disease")',
            '(lecanemab OR donanemab OR aducanumab OR "anti-amyloid" OR "disease modifying therapy") AND ("Alzheimer disease" OR "Alzheimer\'s disease")',
            '"amyloid-related imaging abnormalities" AND ("Alzheimer disease" OR lecanemab OR donanemab)',
            '(donepezil OR rivastigmine OR galantamine OR memantine OR "cholinesterase inhibitor") AND ("Alzheimer disease" OR dementia)',
            '("caregiver burden" OR "cognitive rehabilitation" OR "non-pharmacological intervention") AND ("Alzheimer disease" OR dementia)',
            '"Alzheimer disease" AND "systematic review"',
            '"Alzheimer disease" AND (guideline OR consensus OR recommendation OR "diagnostic criteria")',
        ],
    },

    "general_health": {
        "queries": [
            '"preventive care" OR "preventive medicine" OR "primary prevention"',
            '("primary care" OR "family medicine" OR "general practice") AND (prevention OR screening OR "risk assessment")',
            '("health promotion" OR "public health" OR "population health") AND (prevention OR lifestyle OR screening)',
            '"lifestyle medicine" OR "lifestyle intervention" OR "behavior change"',
            '(exercise OR "physical activity" OR "sedentary behavior") AND ("health outcomes" OR prevention OR mortality)',
            '(diet OR nutrition OR "healthy diet" OR "Mediterranean diet") AND ("health outcomes" OR prevention)',
            '(sleep OR "sleep quality" OR insomnia OR "sleep duration") AND ("health outcomes" OR "cardiometabolic health")',
            '("cardiovascular disease" OR CVD OR "cardiovascular risk") AND (prevention OR "risk assessment" OR lifestyle)',
            '(hypertension OR "high blood pressure") AND (prevention OR management OR lifestyle)',
            '(cholesterol OR dyslipidemia OR "lipid lowering" OR statin) AND (prevention OR "cardiovascular risk")',
            '("type 2 diabetes" OR diabetes OR prediabetes) AND (prevention OR lifestyle OR management)',
            '("metabolic syndrome" OR "cardiometabolic health" OR obesity) AND (prevention OR lifestyle)',
            '(obesity OR overweight OR "weight management" OR BMI) AND (prevention OR lifestyle OR intervention)',
            '(screening OR "early detection") AND (cancer OR "colorectal cancer" OR "breast cancer" OR "lung cancer")',
            '("health screening" OR "preventive screening") AND ("primary care" OR "general population")',
            '("mental health" OR depression OR anxiety) AND ("primary care" OR prevention OR screening OR lifestyle)',
            '(depression OR anxiety) AND (exercise OR sleep OR diet OR "behavioral intervention")',
            '("healthy aging" OR "successful aging") AND (lifestyle OR prevention OR "physical activity" OR nutrition)',
            '(frailty OR sarcopenia OR "functional decline") AND (prevention OR exercise OR nutrition)',
            '("digital health" OR telemedicine OR "mobile health" OR mHealth) AND ("primary care" OR prevention)',
            '("artificial intelligence" OR "machine learning") AND ("primary care" OR "preventive care" OR "risk prediction")',
            '("patient education" OR "health literacy" OR "shared decision making") AND ("primary care" OR prevention)',
            '(adherence OR "medication adherence") AND ("chronic disease" OR "primary care" OR prevention)',
            '"preventive medicine" AND "systematic review"',
            '"preventive care" AND (guideline OR consensus OR recommendation)',
        ],
    },
}

VALID_CATEGORIES = set(SOURCE_GROUPS.keys())


def normalize_title(title: str) -> str:
    """Lowercase + strip punctuation for fuzzy title deduplication."""
    return re.sub(r"[.:\-–]", " ", title.lower()).strip()


def dedupe_records(records: list[dict]) -> list[dict]:
    """Deduplicate by PMCID > PMID > DOI > normalized title (first wins)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for r in records:
        key = r.get("pmcid") or r.get("pmid") or r.get("doi") or normalize_title(r.get("title", ""))
        if key and key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


MANIFEST_FIELDS = [
    "category", "pmid", "pmcid", "doi", "title", "journal", "year",
    "authors", "query", "pmc_url", "txt_path", "license", "word_count",
]
