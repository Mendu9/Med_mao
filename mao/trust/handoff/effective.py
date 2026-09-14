r"""The EFFECTIVE fields of a compiled projection, and where each one came from.

## The defect this exists to remove

Three things used to describe the same projection and none of them was the same
object:

    the extractor        reported which source spans it turned into facts
    the compiler         built the shipped context from `_listed(key) or
                         facts.<field>` — a caller-supplied structured field
                         DISPLACING the extracted one, silently
    `SourceAccounting`   counted a segment `TYPED_FACT` from the FIRST of those
                         and never heard about the second

So a segment stayed in the numerator of `coverage()` after its represented value
had been displaced out of the payload. Measured through the real report boundary
at `ea46e7e`: `Complete heart block`, `Penicillin allergy - anaphylaxis` and
`Warfarin 5mg od` absent from the payload at coverage 1.0000, with the payload
affirmatively stating `Complete`, on a request asking whether donepezil — a
bradycardic drug — is safe.                                          ADV19-1

`00_RULES.md`: *"A transformation may not silently delete clinically material
content and then present the result as equivalent to the original."*

## Why a model and not a sixth predicate

This is the fifth defect in the absent-and-complete class and the first that is
not an exclusion predicate INSIDE the account. The four before it
(`_is_clinical_line`, `_carries_nothing_to_lose`, `_PLACEHOLDER`,
`_MARKDOWN_HEADING`) were each a rule deciding what to leave out, and each was
defeated by an input its author had not looked at. This one is a DISAGREEMENT
between two objects, so a rule in either object cannot reach it:

  - scrubbing or validating the structured fields is the LEAK direction
    (AR18-2, deferred) and does not touch this;
  - removing the override defeats the refusal pathway — `00_RULES` names
    structured fields as the remedy for ambiguity, and `HandoffRefused._WANTED`
    asks for `findings` and `medications (with dose)` by name;
  - merging caller and extracted values narrows the population without making
    the two objects agree, and presents a clinician's correction and the text it
    corrects as two coexisting facts about one patient.

What removes the disagreement is a single description of what the projection
EFFECTIVELY carries, which the resolution, the compiled projection and the
account all read. The override then becomes a fact the account OBSERVES rather
than one it cannot see, and `accounting.py`'s `TYPED_FACT` contract — *"this
segment became a field of the projection, and the projection can be asked to
show it"* — is true by construction.

## The two readings, and why neither is a second opinion

`resolve()` is the BUILD side: it performs the override arithmetic once, with
provenance, and `compiler.case_from_facts` assembles the `ProtectedCaseContext`
out of its values — for SIX of the eight fields.

ADV20-6 / ADV20-3, deferred from Phase 1 and corrected here in P2-1: `vitals`
and `labs` do NOT come from the resolution. `case_from_facts` builds them from
`facts` directly (`compiler.py:148-149`), and `resolve` hardcodes both to a bare
`stated=()` literal (see `resolve`). The control-document half of this was
corrected at Phase 1 close; this docstring was not, because no application code
was modified at that gate — so the file said "out of its values" without
qualification while two fields bypassed the model entirely.

That matters beyond accuracy: `HandoffRefused._WANTED` asks a refused caller for
`vitals` and `labs` BY NAME, and a caller who resupplies either has the key
silently dropped. Two of the seven fields in the refusal advice are INERT. That
is ADV20-5, it is NOT fixed here, and it belongs with the structured-field
schema work — the same place that decides which keys may be SENT is the place to
state which are HONOURED. Fixing the docstring does not fix the field.

`of_case()` is the MEASURE side: it reads those same fields back off the
compiled context — the object `SafeSynthesisContext` is built from field by
field — and that is what the account is re-stated against.

They are not two detectors of one condition, which is the C2 anti-pattern this
project has recorded. They are one model written by one function and read back
off the object it produced, so the account is a function of the payload rather
than of anything upstream of it. Measuring the COMPILED context rather than the
resolution is deliberate: it is the reading that stays correct if a caller
assembles a context some other way.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mao.trust.classes import ProtectedCaseContext
from mao.trust.handoff.extract import ExtractedFacts, age_group_for


class FieldSource(str, Enum):
    """Where the value a projection field carries actually came from."""

    #: The document's own content, carried by the deterministic extractor.
    EXTRACTED = "extracted"
    #: A structured field the caller stated. An APPROVED override: the caller
    #: stating a patient's age is better evidence than a regex finding a
    #: two-digit number near the word "aged", and this is the pathway the
    #: ambiguity refusal points at, so it has to be worth using.
    CALLER_STRUCTURED = "caller_structured"
    #: The field carries nothing.
    ABSENT = "absent"


#: Projection fields the EXTRACTOR can carry a source segment into. These are
#: the field names `extract` records against a span, and the only names
#: `represents` can be asked about affirmatively.
#:
#: `vitals` and `labs` are here although nothing overrides them today. A field
#: that cannot be displaced still has to be ASKED, because the guarantee is
#: about the projection and not about the list of keys someone remembered.
EXTRACT_BACKED = ("age_group", "medications", "findings", "vitals", "labs")

#: Projection fields only a caller populates. Listed so the model describes the
#: whole projection rather than the half that happens to be contested: a reader
#: auditing "what can displace document content" should be able to see that
#: these cannot, rather than infer it from their absence.
CALLER_ONLY = ("conditions", "allergies", "uncertainties")


@dataclass(frozen=True)
class EffectiveField:
    """One field of the compiled projection: what it carries, and from where."""

    name: str
    values: tuple[str, ...] = ()
    source: FieldSource = FieldSource.ABSENT
    #: Extracted values this field does NOT carry. Non-empty only where a
    #: caller's structured field displaced the document's own content, which is
    #: exactly the population ADV19-1 is about.
    displaced: tuple[str, ...] = ()


@dataclass(frozen=True)
class EffectiveProjection:
    """Every field of one compiled projection, with its provenance."""

    fields: tuple[EffectiveField, ...] = ()

    @classmethod
    def resolve(
        cls, facts: ExtractedFacts, structured: dict[str, str] | None = None
    ) -> EffectiveProjection:
        """The BUILD side: perform the override once, and record what it cost.

        Structured fields WIN, unchanged from before — that is an approved
        control decision and the refusal pathway depends on it. What is new is
        that winning is now WRITTEN DOWN, so the account can read it.
        """
        stated = structured or {}
        resolved = [
            _field("age_group", _stated_band(stated, facts), (facts.age_group,)),
            _field("medications", _listed(stated, "medications"), facts.medications),
            _field("findings", _listed(stated, "findings"), facts.findings),
            # Not overridable: built from the extractor only. Stated positively
            # rather than left out, so that making one of them overridable later
            # is a visible edit here rather than a silent widening elsewhere.
            _field("vitals", (), _pairs(dict(facts.vitals))),
            _field("labs", (), _pairs(dict(facts.labs))),
        ]
        resolved.extend(
            _field(name, _listed(stated, name), ()) for name in CALLER_ONLY
        )
        return cls(fields=tuple(resolved))

    @classmethod
    def of_case(
        cls, case: ProtectedCaseContext, facts: ExtractedFacts
    ) -> EffectiveProjection:
        """The MEASURE side: what the COMPILED projection effectively carries.

        Read off the `ProtectedCaseContext` that `SafeSynthesisContext` is built
        from field by field, so this describes the payload rather than anything
        upstream of it.

        `facts` is consulted for PROVENANCE only — to say which extracted values
        a field does not carry. `represents()`, the answer the account depends
        on, reads `values` alone, so the invariant does not depend on the
        extractor's view of itself.
        """
        carried = {
            "age_group": (case.demographics.get("age_group", ""),),
            "medications": case.medications,
            "findings": case.findings,
            "vitals": _pairs(case.vitals),
            "labs": _pairs(case.labs),
            "conditions": case.conditions,
            "allergies": case.allergies,
            "uncertainties": case.uncertainties,
        }
        extracted = {
            "age_group": (facts.age_group,),
            "medications": facts.medications,
            "findings": facts.findings,
            "vitals": _pairs(dict(facts.vitals)),
            "labs": _pairs(dict(facts.labs)),
        }
        return cls(
            fields=tuple(
                _observed(name, values, extracted.get(name, ()))
                for name, values in carried.items()
            )
        )

    def field(self, name: str) -> EffectiveField | None:
        for effective in self.fields:
            if effective.name == name:
                return effective
        return None

    def values_for(self, name: str) -> tuple[str, ...]:
        effective = self.field(name)
        return effective.values if effective is not None else ()

    def source_for(self, name: str) -> FieldSource:
        effective = self.field(name)
        return effective.source if effective is not None else FieldSource.ABSENT

    def represents(self, field: str, value: str) -> bool:
        """Can this projection be asked to show `value` in `field`?

        The one question the account puts to the projection, and the whole of
        the ADV19-1 closure. It reads `values` and nothing else: not the
        extractor, not the structured dict, not whether an override happened.

        Fails CLOSED. An unnamed field, or a field this projection does not
        have, cannot show anything — so a future extractor that carries a span
        into a field nobody added here makes that span `UNRESOLVED` rather than
        silently trusted. Pessimism in the safe direction is the only direction
        this subsystem is allowed to be wrong in.
        """
        if not field or not value:
            return False
        effective = self.field(field)
        if effective is None:
            return False
        return value in effective.values

    def displaced(self) -> tuple[str, ...]:
        """Every extracted value the compiled projection does not carry."""
        return tuple(
            value for effective in self.fields for value in effective.displaced
        )


def _listed(structured: dict[str, str], key: str) -> tuple[str, ...]:
    """A structured field as the projection carries it: `;`-separated values.

    Deduplicated in order. A caller who repeats a value is stating it once, and
    a projection that showed it twice would present one fact as two.
    """
    raw = str(structured.get(key, "") or "")
    return tuple(
        dict.fromkeys(part.strip() for part in raw.split(";") if part.strip())
    )


def _stated_band(structured: dict[str, str], facts: ExtractedFacts) -> tuple[str, ...]:
    """The caller's age as a BAND, or nothing if it is not a number.

    An unparseable age is not an override: it establishes nothing, so the
    extracted band stands and the document's own age span is still carried.
    """
    raw = str(structured.get("age", "") or "").strip()
    if not raw:
        return ()
    try:
        return (age_group_for(int(raw)),)
    except ValueError:
        return ()


def _pairs(mapping: dict[str, str]) -> tuple[str, ...]:
    """A vitals/labs mapping as the values a span can be matched against.

    `HR 36` is carried as `heart_rate=36`, not as its source text, so the
    account and the projection have to agree on one spelling of it. This is it.
    """
    return tuple(f"{name}={value}" for name, value in sorted(mapping.items()))


def _field(
    name: str, stated: tuple[str, ...], extracted: tuple[str, ...]
) -> EffectiveField:
    """One resolved field. The ONE place a structured value displaces content."""
    extracted = tuple(value for value in extracted if value)
    if stated:
        return EffectiveField(
            name=name,
            values=stated,
            source=FieldSource.CALLER_STRUCTURED,
            displaced=tuple(value for value in extracted if value not in stated),
        )
    if extracted:
        return EffectiveField(name=name, values=extracted, source=FieldSource.EXTRACTED)
    return EffectiveField(name=name)


def _observed(
    name: str, values: tuple[str, ...], extracted: tuple[str, ...]
) -> EffectiveField:
    """One field as COMPILED, with its provenance inferred from the facts."""
    values = tuple(value for value in values if value)
    extracted = tuple(value for value in extracted if value)
    missing = tuple(value for value in extracted if value not in values)
    if not values:
        return EffectiveField(name=name, displaced=missing)
    source = (
        FieldSource.EXTRACTED
        if extracted and not missing
        else FieldSource.CALLER_STRUCTURED
    )
    return EffectiveField(name=name, values=values, source=source, displaced=missing)
