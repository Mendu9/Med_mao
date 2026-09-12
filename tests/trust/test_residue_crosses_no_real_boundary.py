r"""O4 — unresolved raw text crosses none of the real boundaries.

## The property

Take a report whose residue is distinctive and unparseable — a line no grammar
in `mao.trust.handoff.extract` carries and no rule in `mao.core.deident` redacts:

    Dictated by Sarah Thompson, medical secretary, extension 4471.

That string must appear in NONE of:

  1. `SafeSynthesisContext.render()`            — the model prompt;
  2. anything reaching `mao.providers.gateway`  — captured at `set_provider`;
  3. the response metadata dict from `clinical_agent._handle_pdf_report`
     — this dict is echoed to the caller, cached in Redis and written to a
       trace, so it is an egress surface;
  4. anything passed to `mao.memory.interface` write or read;
  5. the trace schema record.

This is AR17-2. `SafeSynthesisContext.uncertainties` used to be populated from
`facts.uncovered` — the residue, which is the population selected precisely FOR
being unparseable and therefore the one with the highest residual-identifier
density — and `render()` emitted it verbatim. Measured then: a full personal
name and a contact extension inside the class whose own docstring prohibits raw
or scrubbed report text. The metadata dict carried the same list.

## Why the probes are bound where they are

`00_RULES.md`: "The egress layer must be tested at the real sink or SDK/network
seam, not only at an upstream abstraction."

  - the provider probe binds at `gateway.set_provider`, the last thing before
    the vendor SDK, so it also sees `synthesise_clinical`, which does not go
    through `complete()` at all;
  - the memory probe binds at `set_memory_store`, the sink, not at the node
    that calls it;
  - the metadata probe reads the dict `_handle_pdf_report` actually returns,
    walked RECURSIVELY through nested lists and dicts and stringified, because
    a residue nested three levels down in `sources[0]["snippet"]` is in Redis
    just as surely as one at the top level.

The recorder plumbing is the suite's own (`tests/trust/recorders.py`) — that is
the technique this file is told to reuse. The oracle is not: `_find_everywhere`
below is this file's own recursive walk, and the residue corpus is its own.

## How this oracle is independent

The reference is a string constant this test wrote into the document. Finding
"is this exact string present" needs no model of how the boundary works, and
nothing in `mao/` is asked where the residue went. `TestTheSearchIsNotBroken`
supplies the other half: a string that IS supposed to travel must be found by
the same walk, so "not found anywhere" cannot mean "the walk is broken".
"""
from __future__ import annotations

import base64
import io

import pytest

from mao.memory import interface as memory_interface
from mao.providers import gateway
from mao.trust.classes import InputChannel
from mao.trust.egress.gateway import RequestProtection, protected_request
from mao.trust.handoff.compiler import compile_handoff
from mao.trust.inputs.boundary import protect_channel
from tests.trust.recorders import RecordingMemory, RecordingProvider

# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------

#: Unparseable by every grammar in `extract`, and matched by no rule in
#: `deident` — a sign-off line, which is ordinary at the foot of a clinic letter
#: and is exactly the population `uncertainties` used to be filled from.
RESIDUE = "Dictated by Sarah Thompson, medical secretary, extension 4471."

#: Fragments of the residue that would each be a disclosure on their own.
RESIDUE_FRAGMENTS = ("Sarah Thompson", "extension 4471", "medical secretary")

#: A finding that IS supposed to travel. The non-vacuity control.
CARRIED_LINE = "Donepezil 10mg od commenced."

REPORT_LINES = [
    "MEMORY CLINIC REVIEW",
    "Patient Name: Harold Nkemdirim",
    "MRN: A1234567",
    "Complete heart block, permanent pacemaker in situ.",
    CARRIED_LINE,
    "MMSE 21/30 at review.",
    "Aged 84.",
    RESIDUE,
]
REPORT_TEXT = "\n".join(REPORT_LINES) + "\n"

QUESTION = "Is donepezil safe for this patient?"


def _report_pdf_b64() -> str:
    """The document as a real PDF, drawn with the pair production uses.

    reportlab in, pypdf out — the same round trip `_extract_pdf_text` performs,
    so the text under test is what pypdf actually produces rather than what this
    file imagines it produces.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    height = 780
    for line in REPORT_LINES:
        pdf.drawString(58, height, line)
        height -= 20
    pdf.save()
    return base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------
# The oracle: a recursive walk that stringifies everything it reaches.
# --------------------------------------------------------------------------


def _find_everywhere(value: object, needles: tuple[str, ...]) -> list[str]:
    """Every needle readable anywhere inside `value`, however deeply nested.

    Independent of the implementation: it knows nothing but `str`, `dict`,
    `list` and `repr`. Dataclasses and anything else are stringified, so a
    residue hiding on an attribute of an object in a list in a dict is still
    found.
    """
    found: set[str] = set()

    def walk(item: object, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(item, str):
            for needle in needles:
                if needle in item:
                    found.add(needle)
            return
        if isinstance(item, dict):
            for key, sub in item.items():
                walk(key, depth + 1)
                walk(sub, depth + 1)
            return
        if isinstance(item, (list, tuple, set, frozenset)):
            for sub in item:
                walk(sub, depth + 1)
            return
        if isinstance(item, (int, float, bool, type(None))):
            return
        # Anything else: its own fields, then its repr as a backstop.
        fields = getattr(item, "__dict__", None)
        if fields:
            walk(dict(fields), depth + 1)
        walk(repr(item), depth + 1)

    walk(value)
    return sorted(found)


NEEDLES = (RESIDUE, *RESIDUE_FRAGMENTS)


# --------------------------------------------------------------------------
# Fixtures binding the REAL sinks
# --------------------------------------------------------------------------


@pytest.fixture()
def provider():
    recording = RecordingProvider()
    gateway.set_provider(recording)
    try:
        yield recording
    finally:
        gateway.reset_provider()


@pytest.fixture()
def memory():
    recording = RecordingMemory()
    memory_interface.set_memory_store(recording)
    try:
        yield recording
    finally:
        memory_interface.reset_memory_store()


@pytest.fixture(scope="module")
def handoff():
    with protected_request(RequestProtection(trace_id="o4")):
        protected = protect_channel(
            REPORT_TEXT, InputChannel.REPORT, refuse_ambiguity=False
        )
    return compile_handoff(
        protected.text, question=QUESTION, events=tuple(protected.events)
    )


# --------------------------------------------------------------------------
# Seam 1 — the model prompt
# --------------------------------------------------------------------------


class TestTheResidueIsNotInTheModelPrompt:
    def test_render_carries_none_of_it(self, handoff) -> None:
        rendered = handoff.synthesis_context.render()
        leaked = _find_everywhere(rendered, NEEDLES)
        assert leaked == [], (
            f"unresolved residue reached the model prompt: {leaked}\n{rendered}"
        )

    def test_the_projection_object_carries_none_of_it(self, handoff) -> None:
        """Not merely absent from the rendering — absent from the object.

        `render()` could omit a field the object still holds, and the object is
        what a future caller might serialise.
        """
        leaked = _find_everywhere(handoff.synthesis_context, NEEDLES)
        assert leaked == [], f"residue is on the projection object: {leaked}"

    def test_uncertainties_is_not_the_residue_channel_any_more(self, handoff) -> None:
        """AR17-2 named directly: the field that used to carry it."""
        assert handoff.synthesis_context.uncertainties == (), (
            "uncertainties is populated on a call that stated none, so it is "
            f"being filled from the extractor again: "
            f"{handoff.synthesis_context.uncertainties!r}"
        )

    def test_the_completeness_report_names_a_line_and_no_text(self, handoff) -> None:
        completeness = handoff.synthesis_context.completeness
        assert completeness.complete is False, (
            "the residue line was carried after all — this document no longer "
            "exercises the property"
        )
        assert completeness.unresolved_lines, "no line is named at all"
        assert _find_everywhere(completeness, NEEDLES) == [], (
            "the completeness report carries source text"
        )
        assert _find_everywhere(completeness.describe(), NEEDLES) == []

    def test_the_residue_really_is_the_unresolved_segment(self, handoff) -> None:
        """The probe is aimed at the right thing.

        If the residue were being carried as a finding, its absence from the
        prompt would prove nothing about the residue channel.
        """
        assert any(
            RESIDUE.strip() in text for text in handoff.facts.unresolved_text()
        ), (
            "the residue line is not what the accounting calls unresolved: "
            f"{handoff.facts.unresolved_text()!r}"
        )


# --------------------------------------------------------------------------
# Seam 2 — the provider gateway
# --------------------------------------------------------------------------


class TestTheResidueDoesNotReachTheProvider:
    def test_nothing_sent_to_the_vendor_seam_carries_it(
        self, handoff, provider
    ) -> None:
        from mao.trust.classes import PublicEvidence

        with protected_request(RequestProtection(trace_id="o4-provider")):
            gateway.synthesise_clinical(
                system_prompt="You are a clinical decision support assistant.",
                context=handoff.synthesis_context,
                evidence=(
                    PublicEvidence(
                        evidence_id="e1",
                        text="Cholinesterase inhibitors are bradycardic.",
                        source_id="guideline",
                    ),
                ),
                max_tokens=256,
            )
        assert provider.calls, (
            "nothing reached the provider seam — this probe is vacuous"
        )
        leaked = _find_everywhere(provider.calls, NEEDLES)
        assert leaked == [], f"residue reached the provider: {leaked}"

    def test_the_payload_that_was_authorised_carries_none_of_it(
        self, handoff
    ) -> None:
        """The `ExternalSafePayload` envelope itself, walked whole."""
        from mao.trust.egress.gateway import authorise
        from mao.trust.egress.policy import Destination, EgressPurpose
        from mao.trust.classes import TrustClass

        with protected_request(RequestProtection(trace_id="o4-envelope")):
            envelope = authorise(
                destination=Destination.MODEL_PROVIDER,
                purpose=EgressPurpose.CLINICAL_SYNTHESIS,
                trust_class=TrustClass.SAFE_SYNTHESIS_CONTEXT,
                texts=(handoff.synthesis_context.render(),),
                payload=handoff.synthesis_context,
            )
        leaked = _find_everywhere(envelope, NEEDLES)
        assert leaked == [], f"residue is inside the authorised envelope: {leaked}"


# --------------------------------------------------------------------------
# Seam 3 — the response metadata dict (Redis cache + trace)
# --------------------------------------------------------------------------


class TestTheResponseMetadataCarriesNoResidue:
    """`_handle_pdf_report`'s second return value, walked recursively.

    This dict is echoed into the HTTP response, cached in Redis and written to a
    trace. At `ff34722` it carried the residue as `clinical_uncertainties`.
    """

    @pytest.fixture(scope="class")
    def result(self):
        from mao.agents.clinical_agent import _handle_pdf_report

        recording = RecordingProvider()
        gateway.set_provider(recording)
        try:
            with protected_request(RequestProtection(trace_id="o4-pdf")):
                response, metadata = _handle_pdf_report(
                    QUESTION, {"report_b64": _report_pdf_b64()}, ""
                )
        finally:
            gateway.reset_provider()
        return response, metadata, recording

    def test_the_extraction_actually_happened(self, result) -> None:
        """Non-vacuity: a failed PDF extraction returns a two-key dict and every
        assertion below would pass for the wrong reason."""
        _, metadata, _ = result
        assert metadata.get("mode") == "pdf_report"
        assert "error" not in metadata, f"extraction failed: {metadata!r}"
        assert "clinical_projection_complete" in metadata, (
            f"the completeness account is not in the metadata: {sorted(metadata)}"
        )

    def test_no_residue_anywhere_in_the_metadata(self, result) -> None:
        _, metadata, _ = result
        leaked = _find_everywhere(metadata, NEEDLES)
        assert leaked == [], (
            f"residue is in the dict that is cached and traced: {leaked}\n"
            f"{metadata!r}"
        )

    def test_the_metadata_reports_the_loss_as_counts_and_line_numbers(
        self, result
    ) -> None:
        """The loss is REPORTED — just not by quoting it."""
        _, metadata, _ = result
        assert metadata["clinical_projection_complete"] is False, (
            "the metadata claims a complete projection for a document with an "
            "unresolved line"
        )
        assert metadata["clinical_segments_not_carried"] >= 1
        assert metadata["clinical_lines_not_carried"], (
            "no source line is named, so a clinician cannot find what was lost"
        )
        assert all(
            isinstance(line, int) for line in metadata["clinical_lines_not_carried"]
        ), "a 'line number' that is not a number may be text"

    def test_no_residue_reached_the_provider_on_the_real_agent_path(
        self, result
    ) -> None:
        _, _, recording = result
        assert recording.calls, "the agent path made no provider call at all"
        leaked = _find_everywhere(recording.calls, NEEDLES)
        assert leaked == [], f"residue reached the provider via the agent: {leaked}"

    def test_no_residue_in_the_response_text(self, result) -> None:
        response, _, _ = result
        assert _find_everywhere(response, NEEDLES) == []


# --------------------------------------------------------------------------
# Seam 4 — memory
# --------------------------------------------------------------------------


class TestTheResidueDoesNotReachMemory:
    def test_nothing_written_to_the_memory_sink_carries_it(
        self, handoff, memory
    ) -> None:
        store = memory_interface.get_memory_store()
        store.remember(
            query=QUESTION,
            response=handoff.synthesis_context.render(),
            user_id="o4",
        )
        assert memory.calls, "nothing reached the memory sink — probe is vacuous"
        leaked = _find_everywhere(memory.calls, NEEDLES)
        assert leaked == [], f"residue reached the memory store: {leaked}"

    def test_nothing_read_from_the_memory_sink_carries_it(
        self, handoff, memory
    ) -> None:
        store = memory_interface.get_memory_store()
        store.recall(query=QUESTION, user_id="o4")
        leaked = _find_everywhere(memory.calls, NEEDLES)
        assert leaked == [], f"residue reached the memory read: {leaked}"


# --------------------------------------------------------------------------
# Seam 5 — the trace record
# --------------------------------------------------------------------------


class TestTheTraceRecordCarriesNoResidue:
    def test_the_trace_schema_built_from_the_real_result_is_clean(self) -> None:
        from mao.agents.clinical_agent import _handle_pdf_report
        from mao.api.tracing import build_trace

        recording = RecordingProvider()
        gateway.set_provider(recording)
        try:
            with protected_request(RequestProtection(trace_id="o4-trace")):
                response, metadata = _handle_pdf_report(
                    QUESTION, {"report_b64": _report_pdf_b64()}, ""
                )
        finally:
            gateway.reset_provider()

        trace = build_trace(
            trace_id="o4-trace",
            result={
                "metadata": metadata,
                "intent": "clinical",
                "final_response": response,
                "risk_level": "high",
            },
            latency_ms=12.0,
        )
        leaked = _find_everywhere(trace.to_dict(), NEEDLES)
        assert leaked == [], f"residue is in the trace record: {leaked}"

    def test_the_trace_is_not_empty(self) -> None:
        """Non-vacuity: an all-defaults trace would carry nothing either."""
        from mao.api.tracing import build_trace

        trace = build_trace(
            trace_id="o4-shape",
            result={"metadata": {}, "intent": "clinical"},
            latency_ms=1.0,
        )
        record = trace.to_dict()
        assert record["trace_id"] == "o4-shape"
        assert record["workflow"] == "clinical"
        assert record["policy_version"], "the trace carries no policy version"


# --------------------------------------------------------------------------
# The non-vacuity control for the SEARCH itself
# --------------------------------------------------------------------------


class TestTheSearchIsNotBroken:
    """NON-VACUITY CONTROL.

    Every assertion above is "not found". That is only evidence if the same walk
    DOES find a string which is supposed to be there. The medication line is
    carried by the projection by design, so it must be found at the seams the
    residue must not be found at.
    """

    def test_the_walk_finds_a_carried_finding_in_the_prompt(self, handoff) -> None:
        rendered = handoff.synthesis_context.render()
        assert _find_everywhere(rendered, (CARRIED_LINE,)) == [CARRIED_LINE], (
            "the medication line is NOT in the payload, so 'residue not found' "
            f"may simply mean the search is broken.\n{rendered}"
        )

    def test_the_walk_finds_a_carried_finding_at_the_provider_seam(
        self, handoff, provider
    ) -> None:
        from mao.trust.classes import PublicEvidence

        with protected_request(RequestProtection(trace_id="o4-control")):
            gateway.synthesise_clinical(
                system_prompt="You are a clinical decision support assistant.",
                context=handoff.synthesis_context,
                evidence=(PublicEvidence(evidence_id="e", text="Guideline."),),
                max_tokens=128,
            )
        assert _find_everywhere(provider.calls, (CARRIED_LINE,)) == [CARRIED_LINE], (
            "the walk cannot find a string that IS sent to the provider"
        )

    def test_the_walk_finds_a_string_nested_deep_in_a_structure(self) -> None:
        """The recursion depth is real, not decorative."""
        haystack = {
            "sources": [
                {"snippet": "irrelevant"},
                {"nested": {"deeper": ["x", {"deepest": RESIDUE}]}},
            ]
        }
        assert RESIDUE in _find_everywhere(haystack, NEEDLES)

    def test_the_walk_looks_inside_an_object_and_not_only_at_strings(self) -> None:
        class Envelope:
            def __init__(self) -> None:
                self.payload = {"note": RESIDUE}

        assert RESIDUE in _find_everywhere(Envelope(), NEEDLES)

    def test_the_walk_finds_each_fragment_independently(self) -> None:
        """A partial leak must be caught too, not only the whole sentence."""
        for fragment in RESIDUE_FRAGMENTS:
            assert _find_everywhere(f"...{fragment}...", NEEDLES) == [fragment]

    def test_the_residue_is_genuinely_present_in_the_protected_text(self) -> None:
        """The document really does contain what we are asserting is contained.

        If the scrubber had redacted the sign-off line, the whole file would be
        asserting the absence of something that was never there.
        """
        with protected_request(RequestProtection(trace_id="o4-presence")):
            protected = protect_channel(
                REPORT_TEXT, InputChannel.REPORT, refuse_ambiguity=False
            )
        assert RESIDUE in protected.text, (
            "the boundary redacted the sign-off line, so it is no longer "
            "residue and this file measures nothing"
        )
