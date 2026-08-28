"""arch-L1 — the eval judge must not hard-code a model id or import a provider SDK.

`mao/eval/llm_judge.py` held `_DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"`
and did `from groq import Groq`, bypassing both the registry and the gateway.
The registry records retired ids precisely so they can never be resolved; a
literal id in a module has nothing to catch it when the provider retires it.
"""
from __future__ import annotations

import inspect

from mao.eval import llm_judge


class TestNoLiteralModelIdOrSdkImport:
    def test_no_hardcoded_judge_model_constant(self) -> None:
        assert not hasattr(llm_judge, "_DEFAULT_JUDGE_MODEL")

    def test_no_literal_model_id_in_the_source(self) -> None:
        source = inspect.getsource(llm_judge)
        assert "llama-3.3-70b-versatile" not in source

    def test_no_provider_sdk_import(self) -> None:
        source = inspect.getsource(llm_judge)
        assert "from groq import" not in source
        assert "import groq" not in source

    def test_it_goes_through_the_gateway(self) -> None:
        source = inspect.getsource(llm_judge)
        assert "gateway.complete" in source


class TestItRunsOnTheSafetyJudgeRole:
    def test_the_judge_asks_for_the_safety_judge_role(self, monkeypatch) -> None:
        from mao.providers.registry import ModelRole

        seen: dict[str, object] = {}

        class _Completion:
            text = '{"accuracy": 8, "safety": 9, "notes": "ok"}'

        def _complete(**kwargs: object) -> _Completion:
            seen.update(kwargs)
            return _Completion()

        monkeypatch.setattr(llm_judge.gateway, "complete", _complete)
        llm_judge.judge_response("q", "a", "ctx")
        assert seen["role"] is ModelRole.SAFETY_JUDGE

    def test_a_failure_still_returns_usable_defaults(self, monkeypatch) -> None:
        def _boom(**kwargs: object) -> object:
            raise RuntimeError("provider down")

        monkeypatch.setattr(llm_judge.gateway, "complete", _boom)
        out = llm_judge.judge_response("q", "a two word answer", "ctx")
        assert out["accuracy"] == 0
        assert out["answer_length"] == 4
        assert "judge error" in out["notes"]
