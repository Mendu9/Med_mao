import pytest
from unittest.mock import patch, MagicMock
from mao.eval.nli_checker import check_claim, check_all_claims

def test_entailed_claim():
    mock_enc = MagicMock()
    mock_enc.predict.return_value = [[0.1, 0.1, 0.8]]
    with patch("mao.eval.nli_checker._get_encoder", return_value=mock_enc):
        result = check_claim("Donepezil is a cholinesterase inhibitor.", "Donepezil inhibits acetylcholinesterase.")
    assert result["entailed"] is True
    assert result["score"] == pytest.approx(0.8, abs=0.01)

def test_contradicted_claim():
    mock_enc = MagicMock()
    mock_enc.predict.return_value = [[0.9, 0.05, 0.05]]
    with patch("mao.eval.nli_checker._get_encoder", return_value=mock_enc):
        result = check_claim("There is no cure for AD.", "Aducanumab cures Alzheimer's disease.")
    assert result["entailed"] is False

def test_check_all_returns_list():
    mock_enc = MagicMock()
    mock_enc.predict.return_value = [[0.1, 0.1, 0.8], [0.05, 0.1, 0.85]]
    with patch("mao.eval.nli_checker._get_encoder", return_value=mock_enc):
        results = check_all_claims(["Claim A", "Claim B"], "Background context here.")
    assert len(results) == 2
    assert all("entailed" in r for r in results)

def test_check_claim_has_contradiction_score():
    mock_enc = MagicMock()
    mock_enc.predict.return_value = [[0.7, 0.2, 0.1]]
    with patch("mao.eval.nli_checker._get_encoder", return_value=mock_enc):
        result = check_claim("premise", "claim")
    assert "contradiction_score" in result
    assert result["contradiction_score"] == pytest.approx(0.7, abs=0.01)
