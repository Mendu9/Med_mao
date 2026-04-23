import pytest
from mao.core.token_counter import count_tokens, truncate_to_budget

def test_count_tokens_basic():
    n = count_tokens("Hello world")
    assert n == 2

def test_count_tokens_empty():
    assert count_tokens("") == 0

def test_truncate_to_budget_no_op():
    chunks = ["short text"]
    result = truncate_to_budget(chunks, web=[], history=[], budget=1000)
    assert result["chunks"] == ["short text"]

def test_truncate_to_budget_drops_chunks_first():
    chunks = ["a " * 500, "b " * 500, "c " * 500]
    result = truncate_to_budget(chunks, web=["w " * 100], history=["h " * 100], budget=800)
    assert result["web"] == ["w " * 100]
    assert result["history"] == ["h " * 100]
    assert len(result["chunks"]) < 3

def test_truncate_to_budget_drops_web_second():
    chunks = []
    web = ["w " * 300, "w2 " * 300]
    history = ["h " * 100]
    result = truncate_to_budget(chunks, web=web, history=history, budget=200)
    assert result["history"] == ["h " * 100]
    assert len(result["web"]) < 2

def test_truncate_to_budget_full_cascade():
    chunks = ["c " * 400]
    web = ["w " * 400]
    history = ["h " * 400]
    result = truncate_to_budget(chunks, web=web, history=history, budget=50)
    assert result["chunks"] == []
    assert result["web"] == []
    assert result["history"] == []
