"""Wave 9 / B11 — `has_attachment` was a fixed six-key list.

The policy's own docstring states the rule without qualification: "An attachment
ALWAYS implies patient data and therefore HIGH risk". Wave 5 established a
second claim when it fixed the audio case: "no attachment is silently
discarded". Both were false.

`ATTACHMENT_KEYS` named six keys, so `{"dicom_b64": <scan>}` with a short query
classified LOW — skipping verification, the council, the judge and the
disclaimer — and the upload was silently discarded, with nothing telling the
clinician their scan had not been looked at. That is the Wave 5 audio defect,
one key away.

`mao/api/cache_key.py` had already reached the conclusion and written it down:
"a fixed list can only ever enumerate the inputs someone thought of", and
inverted its default. The policy kept answering from the list. This closes that
gap, so the risk answer and the cache answer now rest on the same principle.

Being wrong by escalating an unrecognised key to HIGH costs a needless
verification pass. Being wrong the other way is the paragraph above.
"""
from __future__ import annotations

import pytest

from mao.safety.policy import RiskLevel, has_attachment, resolve_risk


class TestAnUnrecognisedAttachmentKeyStillCounts:
    @pytest.mark.parametrize(
        "key",
        [
            "dicom_b64",          # the reviewer's example
            "dicom_study_uid",
            "scan_b64",
            "video_b64",
            "waveform_b64",
            "ecg_b64",
            "slide_image_b64",
            "genome_vcf_b64",
            "attachment",
        ],
    )
    def test_an_unknown_key_implies_patient_data(self, key: str) -> None:
        assert has_attachment({key: "<payload>"}) is True

    @pytest.mark.parametrize("key", ["dicom_b64", "scan_b64", "ecg_b64"])
    def test_an_unknown_attachment_resolves_to_high_risk(self, key: str) -> None:
        """The controls that hang off risk — verification, the council, the
        judge, the disclaimer — all key off this one answer."""
        assert resolve_risk({"metadata": {key: "<payload>"}}) is RiskLevel.HIGH

    def test_an_unknown_key_cannot_be_declassified_by_a_stated_level(self) -> None:
        """Rule 0 of `resolve_risk` — an attachment is checked before any
        explicit level. That has to hold for unrecognised keys too, or the
        inversion is defeated by one extra field."""
        state = {"risk_level": "low", "metadata": {"dicom_b64": "<scan>"}}
        assert resolve_risk(state) is RiskLevel.HIGH


class TestKnownHarmlessKeysDoNotEscalate:
    """The inversion must not classify every ordinary request HIGH — that would
    trade one silent failure for a system that cannot answer a plain question."""

    @pytest.mark.parametrize(
        "metadata",
        [
            {},
            {"domain": "alzheimer"},
            {"workflow": "evidence_qa"},
            {"index_version": "v3"},
            {"reranker_id": "bge-base"},
            {"domain": "stroke", "locale": "en-GB"},
        ],
    )
    def test_ordinary_request_metadata_is_not_an_attachment(
        self, metadata: dict
    ) -> None:
        assert has_attachment(metadata) is False

    def test_a_plain_question_is_not_high_risk(self) -> None:
        state = {"intent": "evidence_qa", "metadata": {"domain": "alzheimer"}}
        assert resolve_risk(state) is not RiskLevel.HIGH

    def test_an_empty_value_still_escalates(self) -> None:
        """Wave 11 / ADV-5 — INVERTED. These asserted False.

        "A key present but empty is not an upload" reads reasonably and is the
        wrong test: it makes an empty string decide a safety question. The same
        falsy-elision defect was fixed in `cache_key.py` as N2 in Wave 9 and left
        standing in the policy that N2's reasoning came from.
        """
        assert has_attachment({"dicom_b64": ""}) is True
        assert has_attachment({"image_b64": None}) is True

    def test_a_harmless_key_with_an_empty_value_does_not_escalate(self) -> None:
        assert has_attachment({"domain": ""}) is False
        assert has_attachment({"locale": None}) is False


class TestThePreviouslyKnownKeysStillWork:
    @pytest.mark.parametrize(
        "key",
        ["image_b64", "image_url", "report_b64", "report_path", "audio_b64", "audio_path"],
    )
    def test_every_original_attachment_key_still_escalates(self, key: str) -> None:
        assert has_attachment({key: "<payload>"}) is True
        assert resolve_risk({"metadata": {key: "<payload>"}}) is RiskLevel.HIGH

    def test_absent_metadata_is_not_an_attachment(self) -> None:
        assert has_attachment(None) is False

    def test_unreadable_metadata_is_still_treated_as_an_attachment(self) -> None:
        """Adversarial L-2 — "we cannot tell" is not "nothing was attached"."""
        assert has_attachment("not-a-dict") is True
