"""M4 — needs_decomposition() missed most realistic multi-part clinical questions.

A false negative is not merely one wasted retrieval pass: it collapses
`sub_queries` to `[query]`, so `senior_supervisor_node` scores completeness
against a single coarse question. An answer that covers the drug but silently
omits the side-effect half is then marked `completeness_ok: True`.
"""
from __future__ import annotations

import pytest

from mao.agents.query_decomposer import needs_decomposition

MULTI_PART = [
    "List the FDA-approved treatments for Alzheimer's disease and their common side effects.",
    "Describe the mechanism of action of lecanemab and explain the ARIA monitoring protocol.",
    "What are the diagnostic criteria for vascular dementia and Lewy body dementia?",
    "Summarize the evidence for aducanumab, memantine, and donepezil in early-stage AD.",
    "Should I start memantine or donepezil first for a patient with moderate AD?",
    "What is the thrombolysis window for acute ischaemic stroke? Include the absolute contraindications.",
    "Give me the dosing schedule for donepezil along with the renal adjustment guidance.",
    "Explain amyloid PET imaging and tau PET imaging in AD staging.",
    "Tell me the sensitivity of plasma p-tau217 and the recommended confirmatory test.",
    "What is amyloid and how does tau cause neurodegeneration?",
    "Compare donepezil and rivastigmine.",
]

SINGLE_PART = [
    "What are tau tangles?",
    "Explain the amyloid cascade hypothesis.",
    "What does APOE4 do?",
    "How does neuroinflammation contribute to dementia?",
    "What is the blood-brain barrier?",
    "What causes ischaemic stroke?",
    "Summarise the pathophysiology of Lewy body dementia.",
]


class TestMultiPartIsDetected:
    @pytest.mark.parametrize("query", MULTI_PART)
    def test_multi_part_question_is_decomposed(self, query: str) -> None:
        assert needs_decomposition(query) is True


class TestSinglePartIsNotDecomposed:
    @pytest.mark.parametrize("query", SINGLE_PART)
    def test_single_part_question_skips_the_model_call(self, query: str) -> None:
        assert needs_decomposition(query) is False


class TestClinicalHistoryPhrasingIsNotMisread:
    """M4/L2 — "and is"/"and was" is narrative, not a second question."""

    def test_clinical_history_conjunction_is_not_multi_part(self) -> None:
        query = "The patient was diagnosed with MCI and is now on donepezil — what should we monitor?"
        assert needs_decomposition(query) is False

    def test_past_tense_history_conjunction_is_not_multi_part(self) -> None:
        query = "She presented with aphasia and was started on aspirin, what is the prognosis?"
        assert needs_decomposition(query) is False


class TestEdgeCases:
    @pytest.mark.parametrize("query", ["", "   ", None])
    def test_empty_input_is_not_decomposed(self, query) -> None:
        assert needs_decomposition(query) is False

    def test_two_explicit_questions_are_decomposed(self) -> None:
        assert needs_decomposition("What is tau? What is amyloid?") is True
