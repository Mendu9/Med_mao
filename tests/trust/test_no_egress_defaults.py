r"""A-5 / ADV16-7 root cause: a trust class asserted by a default parameter.

`multimodal_agent.py` called `gateway.complete(...)` with
`purpose=EgressPurpose.IMAGE_ANALYSIS` and NO `trust_class` argument, so it took
the `complete()` default of `SAFE_DERIVED_TEXT` - "de-identified text minted by
the protected input boundary" - for a base64 patient scan that had been through
no boundary at all. Nothing at the call site stated a class; the signature
stated one on its behalf, and it was wrong.

`policy._validate()` gets the fail-closed property of the whole table from
checking DECLARED classes, so one false declaration defeats it, and the defeat
is invisible to every test. `EgressPurpose`'s own docstring already refuses to
carry a default for exactly this reason. This asserts the same of the class.
"""
from __future__ import annotations

import inspect

import pytest

from mao.providers import gateway


class TestNoEgressParameterHasADefault:
    @pytest.mark.parametrize("name", ["purpose", "trust_class"])
    @pytest.mark.parametrize("function", ["complete", "stream"])
    def test_it_must_be_stated_at_every_call_site(
        self, function: str, name: str
    ) -> None:
        parameter = inspect.signature(getattr(gateway, function)).parameters[name]
        assert parameter.default is inspect.Parameter.empty, (
            f"gateway.{function}({name}=...) carries a default, so a call site "
            "that says nothing has one asserted on its behalf. That is the A-5 "
            "mechanism, and it is how a patient scan left the process declared "
            "SAFE_DERIVED_TEXT."
        )

    def test_calling_complete_without_a_trust_class_is_a_type_error(self) -> None:
        """The signature, not a runtime check, is what enforces this - so a
        call site that forgets cannot run at all rather than running wrongly."""
        from mao.providers.registry import ModelRole
        from mao.trust.egress.policy import EgressPurpose

        with pytest.raises(TypeError, match="trust_class"):
            gateway.complete(
                role=ModelRole.ROUTER_FAST,
                messages=[{"role": "user", "content": "hello"}],
                purpose=EgressPurpose.ROUTING,
            )


class TestEverySinkWrapperDeclaresItsClass:
    """The non-model wrappers must not reintroduce the default they replace."""

    def test_no_wrapper_takes_a_trust_class_from_its_caller(self) -> None:
        from mao.trust.egress import sinks

        for name in dir(sinks):
            if not name.startswith("authorise_"):
                continue
            parameters = inspect.signature(getattr(sinks, name)).parameters
            assert "trust_class" not in parameters, (
                f"sinks.{name} lets its caller choose the class. The point of "
                "these wrappers is that the sink declares what it carries."
            )
