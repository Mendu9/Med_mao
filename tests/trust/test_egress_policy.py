r"""The egress contract, asserted at the policy and at the real sink.

Phase 1 exit invariants 11 and 12:

    11. every trust class — ProtectedCaseContext included, not only
        RawSensitiveInput — has a recorded, enforced, testable destination
        policy, and no trust class reaches an unnamed destination by default;
    12. external clinical synthesis receives only SafeSynthesisContext +
        PublicEvidence; RawSensitiveInput, ProtectedCaseContext, raw/scrubbed
        reports, raw history and raw transcripts are rejected at the
        EgressGateway and the real synthesis sink.

"and the real synthesis sink" is the operative clause. A test that patches
`gateway.complete` does not observe the control, it REPLACES it — which is how
a previous PHI assertion came to pass whether or not the control existed. So
every acceptance and rejection here is driven through `gateway.set_provider`,
the last seam before the vendor SDK, and every probe asserts it was not vacuous.
"""
from __future__ import annotations

import pytest

from mao.providers import gateway
from mao.providers.registry import ModelRole
from mao.trust.classes import (
    NEVER_EXTERNAL,
    InputChannel,
    ProtectedCaseContext,
    PublicEvidence,
    RawSensitiveInput,
    SafeEvidenceQuery,
    SafeSynthesisContext,
    TrustClass,
)
from mao.trust.egress.gateway import (
    EgressRefused,
    RequestProtection,
    authorise,
    protected_request,
)
from mao.trust.egress.policy import (
    Destination,
    EgressPurpose,
    allowed_classes,
    is_named,
    named_flows,
)

from .recorders import RecordingProvider


@pytest.fixture
def recorder():
    """A provider bound at the real sink, restored afterwards."""
    recording = RecordingProvider()
    gateway.set_provider(recording)
    try:
        yield recording
    finally:
        gateway.reset_provider()


def _synthesis_context() -> SafeSynthesisContext:
    return SafeSynthesisContext(
        context_id="c1",
        clinical_question="Is it safe to continue donepezil?",
        age_group="older_adult_75_89",
        medications_and_doses=("donepezil 10 mg once daily",),
        clinically_relevant_findings=("bradycardia 48 bpm", "dizziness"),
    )


class TestTheTableSaysWhatThePolicySays:
    """Invariant 11. The document and the table are one decision, stated twice."""

    def test_no_flow_admits_a_class_that_may_never_leave(self) -> None:
        for destination, purpose, classes in named_flows():
            forbidden = classes & NEVER_EXTERNAL
            assert not forbidden, (
                f"{destination.value}/{purpose.value} admits "
                f"{sorted(c.value for c in forbidden)}"
            )

    def test_the_never_external_set_is_exactly_the_two_protected_classes(self) -> None:
        """Pinned. Widening it is a control-plane decision, not a code edit."""
        assert NEVER_EXTERNAL == frozenset(
            {TrustClass.RAW_SENSITIVE_INPUT, TrustClass.PROTECTED_CASE_CONTEXT}
        )

    def test_clinical_synthesis_admits_only_the_typed_projection(self) -> None:
        admitted = allowed_classes(
            Destination.MODEL_PROVIDER, EgressPurpose.CLINICAL_SYNTHESIS
        )
        assert admitted == frozenset({TrustClass.SAFE_SYNTHESIS_CONTEXT}), (
            "the approved M-1 decision names SafeSynthesisContext and nothing "
            f"else for external clinical synthesis; the table admits {admitted}"
        )
        assert TrustClass.SAFE_DERIVED_TEXT not in admitted, (
            "a scrubbed free-text report is a SafeDerivedText. Admitting it "
            "here would make the decision that removed the universal scrubber "
            "from this boundary false again."
        )

    def test_every_destination_and_purpose_pair_is_either_named_or_refused(self) -> None:
        """Fail-closed, exhaustively: the cartesian product has no third state."""
        for destination in Destination:
            for purpose in EgressPurpose:
                if is_named(destination, purpose):
                    continue
                with pytest.raises(EgressRefused, match="no approved flow"):
                    authorise(
                        destination=destination,
                        purpose=purpose,
                        trust_class=TrustClass.SAFE_DERIVED_TEXT,
                    )

    def test_the_named_flows_are_few_enough_to_read(self) -> None:
        """A table nobody can read is a table nobody audits."""
        flows = named_flows()
        assert flows, "no flow is approved at all"
        assert len(flows) <= 20, (
            f"{len(flows)} approved flows. Each one is a decision somebody has "
            "to be able to justify at a review."
        )


class TestProtectedClassesAreRefusedEverywhere:
    """Invariant 11's "ProtectedCaseContext included, not only RawSensitiveInput"."""

    @pytest.mark.parametrize("trust_class", sorted(NEVER_EXTERNAL, key=lambda c: c.value))
    @pytest.mark.parametrize("destination", list(Destination))
    def test_no_destination_accepts_a_protected_class(
        self, trust_class: TrustClass, destination: Destination
    ) -> None:
        for purpose in EgressPurpose:
            with pytest.raises(EgressRefused):
                authorise(
                    destination=destination, purpose=purpose, trust_class=trust_class
                )

    def test_the_protected_types_exist_and_are_distinguishable(self) -> None:
        """A trust class that is not a type cannot be refused by type."""
        raw = RawSensitiveInput(query="Patient Name: Harold Nkemdirim")
        case = ProtectedCaseContext(case_id_internal="x")
        assert raw.trust_class is TrustClass.RAW_SENSITIVE_INPUT
        assert case.trust_class is TrustClass.PROTECTED_CASE_CONTEXT
        assert raw.trust_class is not case.trust_class


class TestTheRealSynthesisSink:
    """Invariant 12, driven through the provider seam."""

    def test_a_valid_projection_is_accepted_and_reaches_the_provider(
        self, recorder: RecordingProvider
    ) -> None:
        completion = gateway.synthesise_clinical(
            system_prompt="You are a clinical assistant.",
            context=_synthesis_context(),
            evidence=(PublicEvidence(evidence_id="e1", text="NICE NG97 says ..."),),
        )
        assert recorder.calls, "the provider was never called — the probe is vacuous"
        assert completion.text
        blob = recorder.text()
        assert "donepezil 10 mg once daily" in blob, "the case facts must arrive"
        assert "NICE NG97" in blob, "the evidence must arrive"

    @pytest.mark.parametrize(
        "payload,label",
        [
            ("Patient Name: Harold Nkemdirim\nMRN: RGT/44219/B\n", "a raw report"),
            ("Patient Name: [NAME]\nMRN: [MRN]\n", "a scrubbed report"),
            ("user: my patient is 78 with bradycardia", "raw chat history"),
            ("So the patient came in last Tuesday and", "a raw transcript"),
        ],
    )
    def test_free_text_is_refused_for_clinical_synthesis(
        self, recorder: RecordingProvider, payload: str, label: str
    ) -> None:
        """There is no parameter through which free text can arrive.

        Asserted twice: `synthesise_clinical` rejects a non-projection, and
        `complete()` refuses the purpose outright. Both are needed — the first
        stops a wrong argument, the second stops a caller routing around the
        typed function entirely.
        """
        with pytest.raises(EgressRefused):
            gateway.synthesise_clinical(
                system_prompt="You are a clinical assistant.",
                context=payload,  # type: ignore[arg-type]
            )
        with pytest.raises(EgressRefused, match="clinical synthesis"):
            gateway.complete(
                role=ModelRole.CLINICAL_SYNTHESIS,
                messages=[{"role": "user", "content": payload}],
                purpose=EgressPurpose.CLINICAL_SYNTHESIS,
            )
        assert not recorder.calls, f"{label} reached the provider"

    def test_the_streaming_route_refuses_it_too(
        self, recorder: RecordingProvider
    ) -> None:
        """Transport is not a safety input.

        `/chat` and `/chat/stream` have disagreed about a refusal before — one
        answered 422 and the other 500 — so the equivalence is asserted rather
        than assumed.
        """
        with pytest.raises(EgressRefused):
            list(
                gateway.stream(
                    role=ModelRole.CLINICAL_SYNTHESIS,
                    messages=[{"role": "user", "content": "Patient Name: X"}],
                    purpose=EgressPurpose.CLINICAL_SYNTHESIS,
                )
            )
        assert not recorder.calls


class TestTheRunScopedIdentifierAssertion:
    """The control that holds for a channel nobody remembered to scrub."""

    @staticmethod
    def _protection() -> RequestProtection:
        protection = RequestProtection(trace_id="t")
        protection.record_identifier(
            "NAME", "Harold Nkemdirim", InputChannel.CHAT_HISTORY
        )
        protection.record_identifier("MRN", "RGT/44219/B", InputChannel.REPORT)
        return protection

    def test_an_identifier_from_any_channel_is_refused_at_the_sink(
        self, recorder: RecordingProvider
    ) -> None:
        with protected_request(self._protection()):
            with pytest.raises(EgressRefused, match="identifier"):
                gateway.complete(
                    role=ModelRole.GENERAL_SYNTHESIS,
                    messages=[
                        {"role": "user", "content": "Summarise for Harold Nkemdirim"}
                    ],
                    purpose=EgressPurpose.GENERAL_SYNTHESIS,
                )
        assert not recorder.calls, "the leaking call reached the provider"

    def test_it_sees_through_a_zero_width_character(
        self, recorder: RecordingProvider
    ) -> None:
        """The assertion compares VISIBLE forms.

        A guard comparing raw strings calls `Harold Nkem<U+034F>dirim` a pass,
        because the identifier is not a literal substring — and that payload is
        exactly as readable to the recipient as the original.
        """
        with protected_request(self._protection()):
            with pytest.raises(EgressRefused, match="identifier"):
                gateway.complete(
                    role=ModelRole.GENERAL_SYNTHESIS,
                    messages=[
                        {"role": "user", "content": "Harold Nkem͏dirim"}
                    ],
                    purpose=EgressPurpose.GENERAL_SYNTHESIS,
                )
        assert not recorder.calls

    def test_it_reads_the_text_half_of_a_multipart_message(
        self, recorder: RecordingProvider
    ) -> None:
        """The vision request is the one most likely to carry a name beside it."""
        with protected_request(self._protection()):
            with pytest.raises(EgressRefused, match="identifier"):
                gateway.complete(
                    role=ModelRole.VISION,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Scan for Harold Nkemdirim"},
                                {"type": "image_url", "image_url": {"url": "data:x"}},
                            ],
                        }
                    ],
                    purpose=EgressPurpose.IMAGE_ANALYSIS,
                )
        assert not recorder.calls

    def test_registered_evidence_is_exempt(self, recorder: RecordingProvider) -> None:
        """A corpus that says `Parkinson` must not refuse a patient called Parkinson.

        The exemption is by REGISTERED SPAN, not by vocabulary: retrieval
        declares what it retrieved and exactly that text is masked. A caller
        cannot claim it.
        """
        protection = RequestProtection(trace_id="t")
        protection.record_identifier("NAME", "Mary Parkinson", InputChannel.REPORT)
        protection.register_evidence(
            "Parkinson's disease and dementia with Lewy bodies overlap clinically."
        )
        with protected_request(protection):
            gateway.complete(
                role=ModelRole.GENERAL_SYNTHESIS,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Evidence: Parkinson's disease and dementia with Lewy "
                            "bodies overlap clinically."
                        ),
                    }
                ],
                purpose=EgressPurpose.GENERAL_SYNTHESIS,
            )
        assert recorder.calls, "a legitimate evidence-bearing call was refused"

    def test_the_full_name_is_still_refused_when_evidence_is_registered(
        self, recorder: RecordingProvider
    ) -> None:
        """The exemption must not become a way to send the identifier."""
        protection = RequestProtection(trace_id="t")
        protection.record_identifier("NAME", "Mary Parkinson", InputChannel.REPORT)
        protection.register_evidence("Parkinson's disease overlaps with DLB.")
        with protected_request(protection):
            with pytest.raises(EgressRefused, match="identifier"):
                gateway.complete(
                    role=ModelRole.GENERAL_SYNTHESIS,
                    messages=[
                        {"role": "user", "content": "The patient Mary Parkinson"}
                    ],
                    purpose=EgressPurpose.GENERAL_SYNTHESIS,
                )
        assert not recorder.calls

    def test_the_assertion_is_a_no_op_outside_a_request(
        self, recorder: RecordingProvider
    ) -> None:
        """A CLI run or an ingestion job has no patient to protect.

        The flow and trust-class checks still apply; only the identifier lookup
        is skipped, because there is nothing to look up.
        """
        gateway.complete(
            role=ModelRole.GENERAL_SYNTHESIS,
            messages=[{"role": "user", "content": "Harold Nkemdirim"}],
            purpose=EgressPurpose.GENERAL_SYNTHESIS,
        )
        assert recorder.calls


class TestTheGuardCanFail:
    """Mutation controls. A probe that cannot fail proves nothing."""

    def test_removing_the_purpose_check_would_be_caught(self) -> None:
        """`purpose` has no default, so a call site cannot omit it."""
        with pytest.raises(TypeError):
            gateway.complete(  # type: ignore[call-arg]
                role=ModelRole.GENERAL_SYNTHESIS,
                messages=[{"role": "user", "content": "x"}],
            )

    def test_widening_the_synthesis_row_is_visible(self) -> None:
        """If the row admitted SafeDerivedText, the M-1 assertion must go red."""
        from mao.trust.egress import policy

        key = (Destination.MODEL_PROVIDER, EgressPurpose.CLINICAL_SYNTHESIS)
        original = policy._ALLOWED[key]
        try:
            policy._ALLOWED[key] = original | {TrustClass.SAFE_DERIVED_TEXT}
            widened = allowed_classes(*key)
            assert TrustClass.SAFE_DERIVED_TEXT in widened, (
                "the mutation did not reach the table it claims to mutate"
            )
        finally:
            policy._ALLOWED[key] = original
        assert TrustClass.SAFE_DERIVED_TEXT not in allowed_classes(*key), (
            "the mutation was not reverted"
        )

    def test_an_evidence_query_is_not_a_synthesis_context(self) -> None:
        """Two projections, two purposes. Neither substitutes for the other."""
        query = SafeEvidenceQuery(query_id="q", search_intent="donepezil safety")
        with pytest.raises(EgressRefused):
            authorise(
                destination=Destination.MODEL_PROVIDER,
                purpose=EgressPurpose.CLINICAL_SYNTHESIS,
                trust_class=query.trust_class,
            )
        authorise(
            destination=Destination.WEB_SEARCH,
            purpose=EgressPurpose.EVIDENCE_SEARCH,
            trust_class=query.trust_class,
        )
