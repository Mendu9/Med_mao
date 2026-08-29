"""Scope item 6 — ToolRegistry.

`01_ARCHITECTURE.md` lists `ToolRegistry` among the required registries, spells
out the fields each tool must declare, and names "convert generic Tool Agent
into ToolRegistry capabilities" as an explicit redesign. `00_RULES.md` requires
tools to be centralised through a typed registry.

None existed: `tool_id` appeared only in `mao/schemas/trace.py`, and
`tool_agent.py` dispatched through a dict literal built inside the function that
used it, with no schema, no trust tier, and no way for anything else in the
system to know what tools exist.
"""
from __future__ import annotations

import pytest

from mao.tools import get_tool, registry
from mao.tools.registry import AuthScope, ReadOrWrite, ToolSpec, TrustTier


class TestEveryToolDeclaresTheMandatedFields:
    @pytest.mark.parametrize("name", registry().names())
    def test_spec_is_complete(self, name: str) -> None:
        spec = get_tool(name)
        assert spec.tool_id
        assert spec.capability
        assert spec.domains
        assert isinstance(spec.read_or_write, ReadOrWrite)
        assert isinstance(spec.trust_tier, TrustTier)
        assert spec.input_schema
        assert spec.output_schema
        assert spec.typical_latency_ms > 0
        assert spec.cost_per_call_usd >= 0

    @pytest.mark.parametrize("name", registry().names())
    def test_spec_declares_what_the_tool_may_reach(self, name: str) -> None:
        """`auth_scope` was absent while PROJECT_STATE claimed every mandated
        field was present. The Wave 6 review found that claim was wrong.

        It is the field a risk policy needs most: the registry could describe
        what a tool costs but not what it can touch, and 00_RULES forbids
        unrestricted SQL, shell, filesystem, secrets and operational DB access.
        """
        assert isinstance(get_tool(name).auth_scope, AuthScope)

    @pytest.mark.parametrize("name", registry().names())
    def test_spec_declares_how_the_tool_fails(self, name: str) -> None:
        """`invoke` returns failures as text by contract, which makes them easy
        to mistake for results. Declaring the modes is what makes them visible.
        """
        modes = get_tool(name).failure_modes
        assert modes, f"{name} declares no failure modes"
        assert all(isinstance(m, str) and m for m in modes)

    def test_no_field_the_architecture_mandates_is_missing(self) -> None:
        """Pins the field set, so the earlier false claim cannot recur silently."""
        mandated = {
            "tool_id", "capability", "domains", "read_or_write", "trust_tier",
            "input_schema", "output_schema", "typical_latency_ms",
            "cost_per_call_usd", "auth_scope", "failure_modes",
        }
        assert mandated <= set(ToolSpec.__dataclass_fields__)

    def test_the_registry_is_not_empty(self) -> None:
        assert registry().names()

    def test_the_tool_agent_tools_are_registered(self) -> None:
        for name in ("web_search", "wikipedia", "calculator"):
            assert name in registry().names()


class TestNoToolIsSilentlyAWriter:
    @pytest.mark.parametrize("name", registry().names())
    def test_phase_one_tools_are_read_only(self, name: str) -> None:
        """Nothing in Phase 1 should be able to mutate anything through a tool."""
        assert get_tool(name).read_or_write is ReadOrWrite.READ


class TestDispatch:
    def test_calculator_runs(self) -> None:
        assert "562.5" in registry().invoke("calculator", "15 * 37.5")

    def test_an_unknown_tool_is_refused_not_guessed(self) -> None:
        with pytest.raises(KeyError):
            get_tool("definitely_not_a_tool")

    def test_invoking_an_unknown_tool_returns_a_usable_message(self) -> None:
        out = registry().invoke("definitely_not_a_tool", "x")
        assert "unknown tool" in out.lower()

    def test_a_failing_tool_returns_text_rather_than_raising(self) -> None:
        """The ReAct loop feeds tool output back to the model; it must be text."""
        assert isinstance(registry().invoke("calculator", "$$$ not math $$$"), str)


class TestRegistrationIsGuarded:
    def test_duplicate_registration_is_refused(self) -> None:
        spec = get_tool("calculator")
        with pytest.raises(ValueError):
            registry().register(spec, spec.handler)


class TestTheAgentDispatchesThroughTheRegistry:
    def test_tool_agent_has_no_private_dispatch_table(self) -> None:
        import inspect

        from mao.agents import tool_agent

        source = inspect.getsource(tool_agent)
        assert '"web_search":' not in source
        assert "mao.tools" in source

    def test_the_prompt_description_is_derived_from_the_registry(self) -> None:
        description = registry().describe_for_prompt()
        for name in registry().names():
            assert name in description
