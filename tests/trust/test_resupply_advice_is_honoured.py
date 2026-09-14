r"""ADV20-5 — the refusal asks for seven fields and honours five.

`HandoffRefused._WANTED` (compiler.py:85-93) names the structured fields that
resolve a refusal. Two of them are inert:

    vitals   `case_from_facts` builds it from `facts` directly (compiler.py:148)
    labs     same (compiler.py:149)
             and `EffectiveProjection.resolve` hardcodes `stated=()` for both,
             so a caller-supplied value is never even considered.

A caller who is refused, reads the advice, and resupplies `vitals` or `labs` has
the key silently dropped and is refused again, identically, with no indication
that those two fields were never going to work.

## Why this is pinned now, and pinned this way

ADV20-1 made the refusal -> structured-resupply loop REACHABLE. Before that fix
the advice reached nobody, because the refusal was an opaque HTTP 500, so two
inert fields in it cost nothing. Making the loop work is exactly what turns this
from latent into user-facing: the product now reliably instructs a clinician to
send two fields it discards.

The remedy is the structured-field schema (AR18-2 + ADV20-5 are one piece of
work from two sides) — the place that constrains which keys may be SENT is the
place to state which are HONOURED. That is P2-12 registry work, and it is not a
thing to patch by special-casing two keys.

`strict=True` for the same reason the R5 pin uses it: when the schema lands and
these fields are honoured, this XPASSes, which pytest reports as a FAILURE.
Whoever closes ADV20-5 is forced back here to delete the marker, so the finding
cannot be closed silently or left behind — and the 422's advice cannot quietly
stay wrong.
"""
from __future__ import annotations

import pytest

from mao.trust.handoff.compiler import _WANTED, case_from_facts
from mao.trust.handoff.extract import extract

_REPORT = (
    "Donepezil 10 mg once daily.\n"
    "Blood pressure 128/76.\n"
    "Atrial fibrillation, rate controlled.\n"
)


def _case(**structured: str):
    return case_from_facts(extract(_REPORT), structured=structured)


def test_the_advice_names_the_fields_this_finding_is_about() -> None:
    """Non-vacuity. If `_WANTED` stops naming vitals/labs the finding is closed
    by removal instead, and this fails first rather than the xfail going stale."""
    assert "vitals" in _WANTED
    assert "labs" in _WANTED


def test_the_honoured_fields_really_are_honoured() -> None:
    """The control. Five of seven DO work, so an xfail below cannot be passing
    because structured resupply is broken outright."""
    case = _case(conditions="bradycardia", medications="bisoprolol 2.5 mg")

    assert "bradycardia" in case.conditions
    assert any("bisoprolol" in medication for medication in case.medications)


@pytest.mark.parametrize("field", ["vitals", "labs"])
@pytest.mark.xfail(
    strict=True,
    reason=(
        "ADV20-5 — `vitals` and `labs` are named in the refusal advice but "
        "hardcoded to stated=() and built from `facts` directly, so a caller "
        "who resupplies them is silently ignored. Fix with the structured-field "
        "schema (AR18-2 + ADV20-5) in P2-12; do not special-case two keys."
    ),
)
def test_a_resupplied_field_named_in_the_advice_is_honoured(field: str) -> None:
    stated = {"vitals": "heart rate 44", "labs": "potassium 2.8"}[field]
    marker = {"vitals": "44", "labs": "2.8"}[field]

    case = _case(**{field: stated})
    carried = " ".join(f"{key} {value}" for key, value in getattr(case, field).items())

    assert marker in carried, (
        f"the refusal advice asks for {field!r} and the value was dropped: {carried!r}"
    )
