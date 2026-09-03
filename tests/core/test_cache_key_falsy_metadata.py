"""Wave 9 / N2 — the metadata digest elided falsy values.

`metadata_digest` inverted its default so that every key the caller sends
participates in the cache key, because a fixed list can only ever enumerate the
answer-changing inputs someone thought of. It then filtered on `if v`, so a key
whose value is falsy dropped back out.

`{"patient_id": 0}` therefore shared a cache entry with no metadata at all — a
much narrower version of the bug the inversion was written to fix, surviving in
one corner of the same function. Presence is what matters for a cache key:
"scoped to patient 0" and "not scoped at all" are different questions and may
have different answers.

`has_attachment` had the identical defect and kept it until Wave 11, where NB3
and ADV-5 recorded it independently: `{"dicom_b64": ""}` classified LOW, so an
empty string decided a safety question. It answers presence now too. The two
questions stay separate — risk versus cache scope, and `mao/api/cache_key.py`
explains why they must not be conflated — they just no longer disagree about
what "present" means.
"""
from __future__ import annotations

import pytest

from mao.api.cache_key import CacheKeyInputs

_FIXED = {
    "user_id": "u1",
    "query": "what does donepezil do?",
    "policy_version": "p1",
    "model_version": "m1",
    "prompt_version": "pr1",
    "index_version": "i1",
}


def _digest_for(metadata: dict) -> str:
    return CacheKeyInputs(metadata=metadata, **_FIXED).metadata_digest()


class TestAFalsyValueStillParticipates:
    @pytest.mark.parametrize(
        "metadata",
        [
            {"patient_id": 0},
            {"encounter_id": 0},
            {"page": 0},
            {"redacted": False},
            {"threshold": 0.0},
            {"tags": []},
            {"extra": {}},
            {"note": ""},
        ],
    )
    def test_it_does_not_collide_with_absent_metadata(self, metadata: dict) -> None:
        assert _digest_for(metadata) != _digest_for({}), (
            f"{metadata} shares a cache entry with no metadata at all"
        )

    def test_zero_and_one_are_different_scopes(self) -> None:
        assert _digest_for({"patient_id": 0}) != _digest_for({"patient_id": 1})

    def test_false_and_true_are_different_scopes(self) -> None:
        assert _digest_for({"redacted": False}) != _digest_for({"redacted": True})

    def test_absent_still_reads_as_none(self) -> None:
        assert _digest_for({}) == "none"


class TestTheDigestIsStillStable:
    def test_the_same_metadata_digests_the_same(self) -> None:
        assert _digest_for({"a": 0, "b": 1}) == _digest_for({"a": 0, "b": 1})

    def test_key_order_does_not_matter(self) -> None:
        assert _digest_for({"b": 1, "a": 0}) == _digest_for({"a": 0, "b": 1})

    def test_an_unserialisable_value_does_not_raise(self) -> None:
        """A caller can send anything, and a cache key that raises turns odd
        metadata into a 500 on the request path."""
        assert _digest_for({"obj": object()})
