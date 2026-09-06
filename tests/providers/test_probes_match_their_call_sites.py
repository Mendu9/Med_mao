"""Wave 9 / B10 — the live probe must exercise what production runs.

`tests/providers/test_live_call_sites.py` is the file the "blocker 3 is closed"
claim rests on: it calls every safety call site at its real role and real token
budget, because a budget that looks fine on paper can still be entirely consumed
by the model's analysis channel, leaving an empty completion that every
downstream control waves through.

It had drifted. Wave 7 moved both supervisors to `EXTRACTION_FAST`; their probes
stayed on `SAFETY_JUDGE`, giving them 2.4-3.1x the ceiling production has, on a
model production never binds. Production was fine — the reviewer ran both at
their real role and both parse — but the *evidence* for a closed blocker was
measuring something else.

The probe now imports `_ROLE` and the budget constants from the modules under
test, so it cannot drift again. These assertions are the belt: they are offline
and unmarked, so they run in the ordinary suite rather than only during a live
run — which is precisely when nobody noticed.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

from mao.providers.registry import ModelRole

#: Advisory reviewers. They inform the answer; they do not hold the safety veto,
#: so binding them to the safety judge would both misprice them and imply an
#: authority they do not have.
_ADVISORY_MODULES = ["domain_supervisor", "senior_supervisor"]


class TestTheSupervisorsBindAnAdvisoryRole:
    @pytest.mark.parametrize("module_name", _ADVISORY_MODULES)
    def test_an_advisory_supervisor_is_not_the_safety_judge(
        self, module_name: str
    ) -> None:
        module = importlib.import_module(f"mao.agents.{module_name}")
        assert module._ROLE is not ModelRole.SAFETY_JUDGE

    @pytest.mark.parametrize("module_name", _ADVISORY_MODULES)
    def test_the_module_exposes_the_role_it_calls_with(self, module_name: str) -> None:
        """`_ROLE` is what the probe imports. If a module inlines its role at the
        call site instead, the probe has to name one — which is the drift."""
        module = importlib.import_module(f"mao.agents.{module_name}")
        assert isinstance(getattr(module, "_ROLE", None), ModelRole)


class TestTheLiveProbeNamesNoLiteralBudgetOrRole:
    """The structural property, asserted against the probe's own source.

    A hardcoded 1024 that happens to match today is the same latent defect as a
    hardcoded role that stopped matching — `TestTheSynthesisCallSites` carried
    both numbers with nothing keeping them in step.
    """

    @staticmethod
    def _probe_source() -> str:
        from tests.providers import test_live_call_sites

        return inspect.getsource(test_live_call_sites)

    @pytest.mark.parametrize("literal", ["1024", "768"])
    def test_no_synthesis_budget_is_hardcoded(self, literal: str) -> None:
        source = self._probe_source()
        assert f", {literal})" not in source, (
            f"the probe hardcodes max_tokens={literal}; import the constant from "
            "the agent that spends it"
        )

    @pytest.mark.parametrize("module_name", _ADVISORY_MODULES)
    def test_the_supervisor_probes_import_the_production_role(
        self, module_name: str
    ) -> None:
        source = self._probe_source()
        assert f"from mao.agents.{module_name} import" in source
        assert "_ROLE" in source, (
            "the supervisor probes must resolve the role from the module under "
            "test, not name one here"
        )


class TestTheBudgetsThemselvesAreCoherent:
    def test_every_probed_budget_is_positive(self) -> None:
        from mao.agents.clinical_agent import _SYNTHESIS_MAX_TOKENS
        from mao.agents.graphrag_agent import (
            _SYNTHESIS_MAX_TOKENS as _GRAPHRAG_SYNTHESIS,
        )

        # `_REPORT_SUMMARY_MAX_TOKENS` is gone with the summariser it budgeted.
        # That call sent the de-identified report to an external model, which
        # the approved M-1 policy does not permit; the report path now builds a
        # SafeSynthesisContext in-process instead. A budget for a call that no
        # longer exists is not a budget to keep coherent.
        for budget in (
            _SYNTHESIS_MAX_TOKENS,
            _GRAPHRAG_SYNTHESIS,
        ):
            assert isinstance(budget, int) and budget > 0
