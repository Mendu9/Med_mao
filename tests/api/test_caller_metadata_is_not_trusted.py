"""Caller-supplied metadata: it is PHI, and it changes the answer.

Two Wave 6 HIGH findings, and they are the same mistake seen from two sides —
`ATTACHMENT_KEYS` was treated as the complete description of what a caller can
put in `metadata`. It is not; it is the list of keys that mean "patient data is
attached". Callers send whatever they like.

adv H-2 — `clinical_node` stripped only `ATTACHMENT_KEYS` and echoed every other
caller key into the response, which is then cached in Redis and returned to the
client. Both shipped frontends send `filename`:

    key=session:b02816259f354505967bb68cc85edea0
      PHI in cached blob: ['Doe_Jane', 'MRN4471023', '1948-03-12']

M5's "persist only de-identified text" held for Postgres and for the provider.
It did not hold for Redis.

adv H-3 — `cache_key.attachment_digest()` folded in only `ATTACHMENT_KEYS`:

    patient_id in key changes it: False
    dicom_study_uid changes it:   False
    attachment under an unknown key changes it: False

so any caller scoping a request by a metadata key outside the six got one cache
entry shared across patients. The reviewer named this exactly: "same defect
class as C1, re-created by fixing C1 with a hard-coded key list".

Scrubbing the echoed values is not the fix. `Doe_Jane_MRN4471023_1948-03-12.pdf`
defeats every labelled rule in the scrubber — there are no labels, and `_`
suppresses the word boundaries the shape rules need. The fix is that a response
carries what the *agent produced*, never what the caller sent.
"""
from __future__ import annotations

import pytest

from mao.api.cache_key import CacheKeyInputs, build_chat_cache_key

PHI_FILENAME = "Doe_Jane_MRN4471023_1948-03-12.pdf"


# ---------------------------------------------------------------------------
# H-2 — the response echoes only what the agent produced
# ---------------------------------------------------------------------------

class TestTheResponseNeverEchoesCallerMetadata:
    @pytest.fixture
    def clinical_result(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        from mao.agents import clinical_agent
        from mao.core.state import make_initial_state

        monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: [])
        monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
        monkeypatch.setattr(clinical_agent, "_call_llm", lambda s, u: "An answer.")

        state = make_initial_state("summarise this", "u1")
        state["metadata"] = {
            "filename": PHI_FILENAME,
            "patient_id": "MRN4471023",
            "modality": "image",
            "report_b64": "",
        }
        return dict(clinical_agent.clinical_node(state))

    def test_the_phi_filename_is_not_echoed(self, clinical_result) -> None:
        assert PHI_FILENAME not in str(clinical_result["metadata"])

    def test_no_caller_supplied_key_is_echoed(self, clinical_result) -> None:
        echoed = set(clinical_result["metadata"])
        assert not (echoed & {"filename", "patient_id", "modality"}), (
            f"caller metadata echoed into the response: {sorted(echoed)}"
        )

    def test_the_agents_own_output_still_reaches_the_client(self, clinical_result) -> None:
        """The strip must not take the provenance the UI needs with it."""
        metadata = clinical_result["metadata"]
        assert metadata["mode"] == "text_question"
        assert "sources" in metadata
        assert "uncertainty_flag" in metadata

    def test_attachment_identifiers_are_still_stripped(self, clinical_result) -> None:
        from mao.safety.policy import ATTACHMENT_KEYS

        assert not (set(clinical_result["metadata"]) & ATTACHMENT_KEYS)


# ---------------------------------------------------------------------------
# H-3 — every metadata key is folded into the cache key
# ---------------------------------------------------------------------------

def _key(metadata: dict) -> str:
    return build_chat_cache_key(
        CacheKeyInputs(user_id="u1", query="what is the stage?", metadata=metadata)
    )


class TestEveryMetadataKeyChangesTheCacheKey:
    @pytest.mark.parametrize(
        "key",
        ["patient_id", "encounter_id", "dicom_study_uid", "some_future_key"],
    )
    def test_a_non_attachment_key_changes_the_key(self, key: str) -> None:
        assert _key({}) != _key({key: "A"}), (
            f"{key} does not participate in the cache key, so two patients "
            "scoped by it share one cached answer"
        )

    @pytest.mark.parametrize(
        "key",
        ["patient_id", "encounter_id", "dicom_study_uid", "some_future_key"],
    )
    def test_two_patients_do_not_collide(self, key: str) -> None:
        assert _key({key: "patient-A"}) != _key({key: "patient-B"})

    def test_an_attachment_under_an_unknown_key_changes_the_key(self) -> None:
        assert _key({}) != _key({"scan_payload": "AAAABBBB"})

    def test_identical_metadata_still_hits(self) -> None:
        """The fix must not make the cache useless."""
        meta = {"patient_id": "A", "image_b64": "xyz"}
        assert _key(meta) == _key(dict(meta))

    def test_key_order_does_not_change_the_key(self) -> None:
        assert _key({"a": "1", "b": "2"}) == _key({"b": "2", "a": "1"})

    def test_the_known_attachment_keys_still_work(self) -> None:
        from mao.safety.policy import ATTACHMENT_KEYS

        for key in sorted(ATTACHMENT_KEYS):
            assert _key({}) != _key({key: "payload"})
            assert _key({key: "payload-A"}) != _key({key: "payload-B"})

    def test_no_raw_metadata_is_recoverable_from_the_key(self) -> None:
        assert PHI_FILENAME not in _key({"filename": PHI_FILENAME})

    def test_unserialisable_metadata_does_not_break_the_key(self) -> None:
        """A caller can send anything; the cache must not 500 on it."""
        assert _key({"when": object()})
