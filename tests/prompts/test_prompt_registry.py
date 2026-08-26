"""Contract tests for the modular prompt registry.

Architecture requires every production prompt to carry name, version, required
variables, and an expected output contract — and to be traceable with responses.
"""
from __future__ import annotations

import pytest

from mao.prompts import PromptRegistry, PromptSpec, get_prompt, registry


class TestPromptSpecContract:
    def test_spec_requires_name_and_version(self) -> None:
        spec = PromptSpec(
            name="demo",
            version="1.0.0",
            template="Hello {who}",
            required_variables=("who",),
            output_contract="text",
        )
        assert spec.name == "demo"
        assert spec.version == "1.0.0"

    def test_render_substitutes_required_variables(self) -> None:
        spec = PromptSpec(
            name="demo",
            version="1.0.0",
            template="Hello {who}",
            required_variables=("who",),
            output_contract="text",
        )
        assert spec.render(who="world") == "Hello world"

    def test_render_rejects_missing_required_variable(self) -> None:
        spec = PromptSpec(
            name="demo",
            version="1.0.0",
            template="Hello {who}",
            required_variables=("who",),
            output_contract="text",
        )
        with pytest.raises(ValueError, match="who"):
            spec.render()

    def test_spec_is_frozen(self) -> None:
        spec = PromptSpec(
            name="demo", version="1.0.0", template="x", required_variables=(), output_contract="text"
        )
        with pytest.raises(Exception):
            spec.version = "2.0.0"  # type: ignore[misc]

    def test_trace_ref_combines_name_and_version(self) -> None:
        spec = PromptSpec(
            name="demo", version="1.2.3", template="x", required_variables=(), output_contract="text"
        )
        assert spec.trace_ref == "demo@1.2.3"


class TestRegistryLookup:
    def test_get_prompt_returns_a_spec(self) -> None:
        spec = get_prompt("router.classify")
        assert isinstance(spec, PromptSpec)

    def test_unknown_prompt_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            get_prompt("no.such.prompt")

    def test_registry_rejects_duplicate_registration(self) -> None:
        local = PromptRegistry()
        spec = PromptSpec(
            name="dup", version="1.0.0", template="x", required_variables=(), output_contract="text"
        )
        local.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            local.register(spec)


class TestProductionPromptsAreRegistered:
    """Every prompt the graph relies on must be resolvable by name."""

    REQUIRED = (
        "router.classify",
        "council.accuracy",
        "council.hallucination",
        "council.safety",
        "domain_supervisor.reconcile",
        "senior_supervisor.completeness",
        "graphrag.synthesis",
        "clinical.synthesis",
        "clinical.extraction",
    )

    @pytest.mark.parametrize("name", REQUIRED)
    def test_prompt_is_registered(self, name: str) -> None:
        assert get_prompt(name).name == name

    @pytest.mark.parametrize("name", REQUIRED)
    def test_prompt_declares_an_output_contract(self, name: str) -> None:
        assert get_prompt(name).output_contract

    def test_all_registered_prompts_have_versions(self) -> None:
        for spec in registry().all():
            assert spec.version, f"{spec.name} has no version"

    def test_no_single_giant_prompt_module(self) -> None:
        """Architecture: 'Do not create one giant prompt file.'"""
        import mao.prompts as pkg
        from pathlib import Path

        pkg_dir = Path(pkg.__file__).parent
        modules = [p for p in pkg_dir.glob("*.py") if p.name not in {"__init__.py", "registry.py"}]
        assert len(modules) >= 3, "prompts must be split across capability modules"
