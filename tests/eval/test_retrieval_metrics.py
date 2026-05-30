"""
Tests for mao/eval/retrieval_metrics.py — pure math functions only.
No ChromaDB, no Groq, no DB required.
"""
import pytest
from mao.eval.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
    _is_specific_question,
    _soft_match_hit,
)


# ---------------------------------------------------------------------------
# compute_retrieval_metrics
# ---------------------------------------------------------------------------

def test_perfect_hit_at_rank_1():
    result = compute_retrieval_metrics(
        retrieved_ids=["chunk_a", "chunk_b", "chunk_c"],
        relevant_ids=["chunk_a"],
        k=3,
    )
    assert result["precision_at_k"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["recall_at_k"] == pytest.approx(1.0, abs=1e-4)
    assert result["reciprocal_rank"] == pytest.approx(1.0, abs=1e-4)


def test_relevant_at_rank_2():
    result = compute_retrieval_metrics(
        retrieved_ids=["chunk_x", "chunk_a", "chunk_b"],
        relevant_ids=["chunk_a"],
        k=3,
    )
    assert result["reciprocal_rank"] == pytest.approx(0.5, abs=1e-4)
    assert result["precision_at_k"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["recall_at_k"] == pytest.approx(1.0, abs=1e-4)


def test_no_relevant_in_top_k():
    result = compute_retrieval_metrics(
        retrieved_ids=["chunk_x", "chunk_y", "chunk_z"],
        relevant_ids=["chunk_a"],
        k=3,
    )
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["f1_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_empty_relevant_ids():
    result = compute_retrieval_metrics(
        retrieved_ids=["chunk_a", "chunk_b"],
        relevant_ids=[],
        k=2,
    )
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["f1_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_empty_retrieved_ids():
    result = compute_retrieval_metrics(
        retrieved_ids=[],
        relevant_ids=["chunk_a"],
        k=5,
    )
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_k_larger_than_retrieved():
    result = compute_retrieval_metrics(
        retrieved_ids=["chunk_a", "chunk_b"],
        relevant_ids=["chunk_a"],
        k=5,
    )
    assert result["precision_at_k"] == pytest.approx(0.2, abs=1e-4)
    assert result["recall_at_k"] == pytest.approx(1.0, abs=1e-4)
    assert result["reciprocal_rank"] == pytest.approx(1.0, abs=1e-4)


def test_multiple_relevant_ids():
    result = compute_retrieval_metrics(
        retrieved_ids=["chunk_a", "chunk_b", "chunk_c", "chunk_d", "chunk_e"],
        relevant_ids=["chunk_a", "chunk_c", "chunk_e"],
        k=5,
    )
    assert result["precision_at_k"] == pytest.approx(3 / 5, abs=1e-4)
    assert result["recall_at_k"] == pytest.approx(1.0, abs=1e-4)
    assert result["reciprocal_rank"] == pytest.approx(1.0, abs=1e-4)


def test_f1_formula():
    result = compute_retrieval_metrics(
        retrieved_ids=["chunk_a", "chunk_x", "chunk_y", "chunk_z", "chunk_b"],
        relevant_ids=["chunk_a", "chunk_b"],
        k=5,
    )
    assert result["precision_at_k"] == pytest.approx(2 / 5, abs=1e-4)
    assert result["recall_at_k"] == pytest.approx(1.0, abs=1e-4)
    expected_f1 = 2 * (2 / 5) * 1.0 / ((2 / 5) + 1.0)
    assert result["f1_at_k"] == pytest.approx(expected_f1, abs=1e-4)


def test_k_equals_zero():
    result = compute_retrieval_metrics(
        retrieved_ids=["chunk_a"],
        relevant_ids=["chunk_a"],
        k=0,
    )
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_retrieval_metrics
# ---------------------------------------------------------------------------

def test_aggregate_empty():
    result = aggregate_retrieval_metrics([])
    assert result["mrr"] == 0.0
    assert result["mean_precision_at_k"] == 0.0
    assert result["mean_recall_at_k"] == 0.0
    assert result["mean_f1_at_k"] == 0.0


def test_aggregate_single():
    per_query = [{"precision_at_k": 0.4, "recall_at_k": 1.0, "f1_at_k": 0.57, "reciprocal_rank": 1.0}]
    result = aggregate_retrieval_metrics(per_query)
    assert result["mrr"] == pytest.approx(1.0, abs=1e-4)
    assert result["mean_precision_at_k"] == pytest.approx(0.4, abs=1e-4)
    assert result["mean_recall_at_k"] == pytest.approx(1.0, abs=1e-4)


def test_aggregate_multiple():
    per_query = [
        {"precision_at_k": 0.2, "recall_at_k": 1.0, "f1_at_k": 0.33, "reciprocal_rank": 1.0},
        {"precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0, "reciprocal_rank": 0.0},
        {"precision_at_k": 0.4, "recall_at_k": 0.5, "f1_at_k": 0.44, "reciprocal_rank": 0.5},
    ]
    result = aggregate_retrieval_metrics(per_query)
    assert result["mrr"] == pytest.approx((1.0 + 0.0 + 0.5) / 3, abs=1e-4)
    assert result["mean_precision_at_k"] == pytest.approx((0.2 + 0.0 + 0.4) / 3, abs=1e-4)


# ---------------------------------------------------------------------------
# _is_specific_question
# ---------------------------------------------------------------------------

_BIOMEDICAL_CHUNK = (
    "Metformin reduces hepatic glucose production by activating AMPK pathway. "
    "Patients with type 2 diabetes showed significant HbA1c reduction after 12 weeks "
    "of metformin therapy compared to placebo. The mechanism involves phosphorylation "
    "of ACC and inhibition of gluconeogenesis enzymes in the liver."
)


def test_specific_question_passes():
    """A question referencing domain terms (metformin, AMPK, gluconeogenesis) should pass."""
    question = "How does metformin activate the AMPK pathway to reduce gluconeogenesis?"
    assert _is_specific_question(question, _BIOMEDICAL_CHUNK) is True


def test_generic_question_fails_no_domain_terms():
    """A generic opener with no domain-specific words should be rejected."""
    question = "What are the effects shown in this study?"
    assert _is_specific_question(question, _BIOMEDICAL_CHUNK) is False


def test_generic_question_with_one_domain_term_fails():
    """Only 1 domain word shared — must have at least 2."""
    question = "What does metformin do?"
    # "metformin" is 1 domain word — len > 4 and in chunk — but only 1, should fail
    assert _is_specific_question(question, _BIOMEDICAL_CHUNK) is False


def test_question_with_two_domain_terms_passes():
    """Exactly 2 domain-specific words from the chunk — should pass."""
    question = "What is the relationship between metformin therapy and HbA1c reduction?"
    assert _is_specific_question(question, _BIOMEDICAL_CHUNK) is True


def test_empty_question_fails():
    assert _is_specific_question("", _BIOMEDICAL_CHUNK) is False


def test_empty_chunk_fails():
    assert _is_specific_question("What does metformin do to AMPK phosphorylation?", "") is False


# ---------------------------------------------------------------------------
# _soft_match_hit
# ---------------------------------------------------------------------------

_RELEVANT_TEXT = (
    "Insulin resistance in adipose tissue leads to increased lipolysis and elevated "
    "free fatty acids in circulation. This contributes to hepatic lipid accumulation "
    "and worsens hyperglycemia in type 2 diabetes patients."
)


def test_soft_match_identical_text():
    """Same text should always be a hit."""
    chunk = {"text": _RELEVANT_TEXT}
    assert _soft_match_hit(chunk, _RELEVANT_TEXT) is True


def test_soft_match_high_overlap_passes():
    """~70% of words shared — should pass default 0.60 threshold."""
    # Take the first sentence and add two extra words
    similar_text = (
        "Insulin resistance in adipose tissue leads to increased lipolysis and elevated "
        "free fatty acids. This contributes to hepatic lipid accumulation and worsens "
        "hyperglycemia in type 2 diabetes."
    )
    chunk = {"text": similar_text}
    assert _soft_match_hit(chunk, _RELEVANT_TEXT) is True


def test_soft_match_low_overlap_fails():
    """Completely unrelated text should not match."""
    unrelated = (
        "The role of CRISPR-Cas9 genome editing in oncology has expanded rapidly. "
        "Tumor suppressor mutations drive malignant transformation across multiple cancer types."
    )
    chunk = {"text": unrelated}
    assert _soft_match_hit(chunk, _RELEVANT_TEXT) is False


def test_soft_match_empty_chunk_fails():
    chunk = {"text": ""}
    assert _soft_match_hit(chunk, _RELEVANT_TEXT) is False


def test_soft_match_empty_relevant_text_fails():
    chunk = {"text": _RELEVANT_TEXT}
    assert _soft_match_hit(chunk, "") is False


def test_soft_match_missing_text_key_fails():
    """Chunk dict without 'text' key should not crash and return False."""
    chunk = {"id": "chunk_123", "metadata": {}}
    assert _soft_match_hit(chunk, _RELEVANT_TEXT) is False


def test_soft_match_custom_threshold():
    """A low threshold of 0.10 allows loose matches; strict 0.60 rejects them."""
    loosely_related = (
        "Adipose tissue dysfunction and hepatic lipid deposition are common metabolic findings."
    )
    chunk = {"text": loosely_related}
    assert _soft_match_hit(chunk, _RELEVANT_TEXT, threshold=0.10) is True
    assert _soft_match_hit(chunk, _RELEVANT_TEXT, threshold=0.60) is False
