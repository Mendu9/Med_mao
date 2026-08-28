"""Scope item 5 — the prompt registry must be *consumed*, not merely populated.

The architecture review's H1: `clinical.extraction` was registered, and the
prompt test asserted its registration, but no code ever read it. The agent ran
inline text that had already diverged from the registered spec on a
de-identification instruction. A registry nothing reads is documentation, not
architecture.

So the test is not "is it registered" but "does any production module still
carry prompt text of its own".
"""
from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil

import pytest

import mao.agents
from mao.prompts import registry

# Module-level string constants that are legitimately not prompts.
_ALLOWED = {
    # A tool schema, assembled into the registered `tool.react` prompt at call
    # time rather than duplicating the tool list inside the registry.
    "mao.agents.tool_agent": {"_TOOLS_DESCRIPTION"},
    # A canned reply shown directly to the user. The chitchat node makes no LLM
    # call at all, so this is UI copy, not prompt text — nothing can diverge
    # from a registered spec because nothing sends it to a model.
    "mao.agents.chitchat_agent": {"_DEFAULT_REPLY"},
}

# A string constant this long is prompt text, not a label or a format fragment.
_PROMPT_LIKE_CHARS = 120


def _agent_modules() -> list[str]:
    return [
        f"mao.agents.{m.name}"
        for m in pkgutil.iter_modules(mao.agents.__path__)
        if not m.name.startswith("_")
    ]


def _module_level_string_constants(module) -> dict[str, str]:
    tree = ast.parse(inspect.getsource(module))
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found[target.id] = node.value.value
    return found


class TestNoAgentCarriesItsOwnPromptText:
    @pytest.mark.parametrize("module_name", _agent_modules())
    def test_agent_holds_no_inline_prompt_constant(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        allowed = _ALLOWED.get(module_name, set())
        offenders = {
            name: value
            for name, value in _module_level_string_constants(module).items()
            if len(value) >= _PROMPT_LIKE_CHARS and name not in allowed
        }
        assert not offenders, (
            f"{module_name} holds inline prompt text: {sorted(offenders)}. "
            "Register it and read it through get_prompt()."
        )


class TestEveryRegisteredPromptIsRead:
    """A registered prompt nothing consumes is the failure mode H1 described."""

    def test_every_prompt_name_appears_in_production_code(self) -> None:
        import pathlib

        root = pathlib.Path(inspect.getfile(mao.agents)).parent.parent
        sources = "\n".join(
            p.read_text(encoding="utf-8")
            for p in root.rglob("*.py")
            if "prompts" not in p.parts and "__pycache__" not in p.parts
        )
        unread = [name for name in registry().names() if f'"{name}"' not in sources]
        assert not unread, f"registered but never consumed: {unread}"


class TestTheClinicalPromptsAreTheDivergenceCase:
    def test_extraction_prompt_is_consumed(self) -> None:
        from mao.agents import clinical_agent

        assert "clinical.extraction" in inspect.getsource(clinical_agent)

    def test_synthesis_prompt_is_consumed(self) -> None:
        from mao.agents import clinical_agent

        assert "clinical.synthesis" in inspect.getsource(clinical_agent)
