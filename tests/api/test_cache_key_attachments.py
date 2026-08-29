"""C1 — the cache key must see every attachment the safety policy calls patient data.

The audited defect: `mao/api/cache_key.py` kept its own four-key attachment
tuple, omitting `audio_b64`/`audio_path`. Because `/chat` consults the cache
*before* `graph.invoke`, an audio request could be served from cache without the
risk gate, verification, the council or the disclaimer ever running — and two
different patients' voice samples produced an identical key.

The key list must therefore have exactly one definition, in the safety policy.
"""
from __future__ import annotations

import inspect

import pytest

from mao.api import cache_key as cache_key_mod
from mao.api.cache_key import CacheKeyInputs, build_chat_cache_key
from mao.safety.policy import ATTACHMENT_KEYS


def _key(metadata: dict, query: str = "does this recording suggest decline?") -> str:
    return build_chat_cache_key(
        CacheKeyInputs(user_id="u1", query=query, metadata=metadata)
    )


class TestEveryPolicyAttachmentChangesTheKey:
    @pytest.mark.parametrize("key_name", sorted(ATTACHMENT_KEYS))
    def test_attachment_presence_changes_the_key(self, key_name: str) -> None:
        assert _key({}) != _key({key_name: "payload-a"})

    @pytest.mark.parametrize("key_name", sorted(ATTACHMENT_KEYS))
    def test_two_different_payloads_do_not_collide(self, key_name: str) -> None:
        """Patient A and Patient B must never share a cache entry."""
        assert _key({key_name: "patient-a"}) != _key({key_name: "patient-b"})


class TestAudioSpecifically:
    """The exact defect C1 described, pinned so it cannot return."""

    def test_audio_b64_is_visible_to_the_cache(self) -> None:
        assert _key({}) != _key({"audio_b64": "<voice sample>"})

    def test_two_patients_voice_samples_differ(self) -> None:
        assert _key({"audio_b64": "patient-a"}) != _key({"audio_b64": "patient-b"})

    def test_audio_path_is_visible_to_the_cache(self) -> None:
        assert _key({}) != _key({"audio_path": "/uploads/a.wav"})


class TestTheKeyListHasOneDefinition:
    def test_cache_key_module_does_not_relist_attachment_keys(self) -> None:
        source = inspect.getsource(cache_key_mod)
        assert '"image_b64"' not in source, (
            "cache_key.py re-lists attachment keys instead of importing "
            "ATTACHMENT_KEYS from mao.safety.policy"
        )

    def test_cache_key_module_imports_the_policy_constant(self) -> None:
        source = inspect.getsource(cache_key_mod)
        assert "ATTACHMENT_KEYS" in source


class TestPromptVersionInvalidatesTheCache:
    """The architecture requires model *and prompt* version in the key.

    A prompt edit changes the answer; without the prompt version in the key the
    old answer is served indefinitely.
    """

    def test_inputs_carry_a_prompt_version(self) -> None:
        assert CacheKeyInputs(user_id="u", query="q").prompt_version

    def test_changing_the_prompt_version_changes_the_key(self) -> None:
        a = build_chat_cache_key(
            CacheKeyInputs(user_id="u", query="q", prompt_version="pv-1")
        )
        b = build_chat_cache_key(
            CacheKeyInputs(user_id="u", query="q", prompt_version="pv-2")
        )
        assert a != b


class TestAllMetadataChangesTheKey:
    """Inverted in Wave 7 — this class asserted adversarial finding H-3.

    It required non-attachment metadata to be invisible to the cache key, on the
    reasoning that only attachments change an answer. `patient_id`,
    `encounter_id` and a DICOM study UID are all non-attachment metadata, and a
    caller scoping a request by any of them got one cached answer shared across
    patients. `/chat` consults this cache *before* `graph.invoke`, so that hit
    also skips the risk gate, the council, the judge and the disclaimer.

    Cache efficiency for a key like `ui_theme` is not worth a mechanism that
    cannot tell `ui_theme` from `patient_id`. The full contract lives in
    tests/api/test_caller_metadata_is_not_trusted.py.
    """

    def test_an_incidental_key_changes_the_key(self) -> None:
        assert _key({}) != _key({"ui_theme": "dark"})

    def test_identical_metadata_still_shares_a_key(self) -> None:
        """The cache must still work for genuinely identical requests."""
        assert _key({"ui_theme": "dark"}) == _key({"ui_theme": "dark"})
