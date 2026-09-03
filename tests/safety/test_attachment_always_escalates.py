"""An attachment implies HIGH risk — including when something claims otherwise.

Raised by both exit-gate reviews (architecture MEDIUM, adversarial L-1).
`SafetyPolicy.risk_for` states the rule without qualification:

    "An attachment (MRI image, patient report) always implies patient data
     and therefore HIGH risk, regardless of the routed intent."

`resolve_risk` did not implement it. An explicit `risk_level` was honoured on an
exact match *before* attachments were considered, so:

    {"risk_level": "low", "metadata": {"image_b64": ...}}
        -> LOW, raw_stream_allowed=True

LOW is the only level permitted to skip output verification and the clinical
disclaimer, so that one state defeats both streaming gates at once, and skips
the council, the judge and the NLI gate with it.

The adversarial reviewer graded it LOW because it is unreachable today —
`risk_gate_node` is the only writer of `risk_level` and it escalates on
attachment. That is an argument about the current call graph, not about the
resolver, and the resolver is the shared one every control consults. It is
cheaper to make the rule true than to keep proving nothing reaches it.

Also closes adversarial L-2: metadata that is present but unreadable now
escalates rather than resolving to "no attachment". A missing metadata object
means "nothing was attached"; a malformed one means "we cannot tell", and those
must not resolve the same way on the path to a clinical answer.
"""
from __future__ import annotations

import pytest

from mao.safety.policy import RiskLevel, has_attachment, resolve_risk

ATTACHMENTS = [
    "image_b64",
    "image_url",
    "report_b64",
    "report_path",
    "audio_b64",
    "audio_path",
]


class TestAnExplicitLowCannotOverrideAnAttachment:
    @pytest.mark.parametrize("key", ATTACHMENTS)
    def test_explicit_low_with_an_attachment_resolves_high(self, key: str) -> None:
        risk = resolve_risk({"risk_level": "low", "metadata": {key: "payload"}})
        assert risk is RiskLevel.HIGH

    @pytest.mark.parametrize("key", ATTACHMENTS)
    def test_the_streaming_gate_is_closed_for_it(self, key: str) -> None:
        from mao.api.streaming import may_stream_raw_tokens

        assert not may_stream_raw_tokens(
            {"risk_level": "low", "metadata": {key: "payload"}}
        )

    @pytest.mark.parametrize("key", ATTACHMENTS)
    def test_verification_is_required_for_it(self, key: str) -> None:
        from mao.safety.policy import get_policy

        risk = resolve_risk({"risk_level": "low", "metadata": {key: "payload"}})
        assert get_policy().requires_verification(risk)
        assert get_policy().requires_clinical_disclaimer(risk)

    def test_explicit_standard_with_an_attachment_also_escalates(self) -> None:
        risk = resolve_risk(
            {"risk_level": "standard", "metadata": {"image_b64": "payload"}}
        )
        assert risk is RiskLevel.HIGH


class TestGenuineLowRiskStillWorks:
    def test_chitchat_without_an_attachment_is_low(self) -> None:
        assert resolve_risk({"risk_level": "low", "metadata": {}}) is RiskLevel.LOW

    def test_missing_metadata_is_not_treated_as_an_attachment(self) -> None:
        assert resolve_risk({"risk_level": "low"}) is RiskLevel.LOW

    def test_none_metadata_is_not_treated_as_an_attachment(self) -> None:
        assert resolve_risk({"risk_level": "low", "metadata": None}) is RiskLevel.LOW

    def test_an_empty_attachment_KEY_still_escalates(self) -> None:
        """Wave 11 / NB3 — INVERTED. This asserted LOW for `{"image_b64": ""}`.

        Classification was being decided by the value's truthiness, so a scan
        whose payload was lost in serialisation resolved exactly like a request
        that never carried one. The key is the declaration; escalating costs one
        needless safety chain, and not escalating skips every control that
        patient data requires.
        """
        assert resolve_risk({"risk_level": "low", "metadata": {"image_b64": ""}}) is (
            RiskLevel.HIGH
        )

    def test_a_harmless_key_with_an_empty_value_stays_low(self) -> None:
        assert resolve_risk({"risk_level": "low", "metadata": {"domain": ""}}) is (
            RiskLevel.LOW
        )


class TestUnreadableMetadataEscalates:
    """adv L-2 — 'we cannot tell' must not resolve like 'nothing attached'."""

    @pytest.mark.parametrize("malformed", ["a string", 42, ["image_b64"], object()])
    def test_a_non_dict_metadata_object_escalates(self, malformed: object) -> None:
        assert resolve_risk({"risk_level": "low", "metadata": malformed}) is (
            RiskLevel.HIGH
        )

    @pytest.mark.parametrize("malformed", ["a string", 42, ["image_b64"]])
    def test_has_attachment_reports_it_as_unreadable(self, malformed: object) -> None:
        assert has_attachment(malformed) is True

    def test_absent_metadata_is_still_false(self) -> None:
        assert has_attachment(None) is False
        assert has_attachment({}) is False
