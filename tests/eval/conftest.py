import json
import pytest
from pathlib import Path
from unittest.mock import patch

_GOLDEN_PATH = Path(__file__).parent / "golden" / "dataset.json"
_CHUNKS_PATH = Path(__file__).parent / "fixtures" / "chunks.json"


def load_golden(filter_agent: bool = False) -> list[dict]:
    entries = json.loads(_GOLDEN_PATH.read_text())
    if filter_agent:
        return [e for e in entries if e.get("expected_agent") and e.get("query")]
    return entries


def _all_chunks() -> dict:
    return json.loads(_CHUNKS_PATH.read_text())


@pytest.fixture(params=load_golden())
def golden_entry(request):
    return request.param


def fixture_chunks_for(query_id: str) -> list[dict]:
    return _all_chunks().get(query_id, [])


@pytest.fixture
def mock_retrieve():
    chunks_by_id = _all_chunks()

    def _fake_retrieve(query: str, domain: str = "alzheimer", top_k: int = 5) -> list[dict]:
        for entry in load_golden():
            if entry["query"] == query:
                return chunks_by_id.get(entry["id"], [])
        return []

    with patch("mao.rag.retriever.retrieve", side_effect=_fake_retrieve) as m:
        yield m


@pytest.fixture
def mock_llm():
    def _fake_invoke(messages, **kwargs):
        class _Msg:
            content = "VERDICT: PASS. Deterministic mock response for testing."

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()

    with patch("mao.core.llm.get_client") as mock_client:
        mock_client.return_value.chat.completions.create.side_effect = _fake_invoke
        yield mock_client
