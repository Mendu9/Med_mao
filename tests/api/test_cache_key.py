"""Cache keys must account for everything that can change the answer.

P1-4: the audited key was md5(f"{user_id}:{query}") only. It ignored chat
history, attachments, and every version dimension — so two different MRI images
with the same caption returned the same cached clinical answer.
"""
from __future__ import annotations

from mao.api.cache_key import CacheKeyInputs, build_chat_cache_key


def _inputs(**kw) -> CacheKeyInputs:
    base = {
        "user_id": "u1",
        "query": "what is tau?",
        "chat_history": [],
        "metadata": {},
        "workflow": "evidence_qa",
    }
    base.update(kw)
    return CacheKeyInputs(**base)


class TestKeyStability:
    def test_identical_inputs_produce_identical_keys(self) -> None:
        assert build_chat_cache_key(_inputs()) == build_chat_cache_key(_inputs())

    def test_key_is_a_stable_digest_not_a_salted_builtin_hash(self) -> None:
        """Must be reproducible across processes (no builtin hash())."""
        key = build_chat_cache_key(_inputs())
        assert key.startswith("query:")
        assert len(key) > len("query:") + 32


class TestAnswerChangingDimensions:
    def test_different_query_changes_the_key(self) -> None:
        assert build_chat_cache_key(_inputs()) != build_chat_cache_key(
            _inputs(query="what is amyloid?")
        )

    def test_different_user_changes_the_key(self) -> None:
        assert build_chat_cache_key(_inputs()) != build_chat_cache_key(_inputs(user_id="u2"))

    def test_chat_history_changes_the_key(self) -> None:
        assert build_chat_cache_key(_inputs()) != build_chat_cache_key(
            _inputs(chat_history=[{"role": "user", "content": "earlier turn"}])
        )

    def test_different_attachment_changes_the_key(self) -> None:
        """The headline P1-4 failure: two MRI images, same caption."""
        a = build_chat_cache_key(_inputs(metadata={"image_b64": "AAAA"}))
        b = build_chat_cache_key(_inputs(metadata={"image_b64": "BBBB"}))
        assert a != b

    def test_different_report_attachment_changes_the_key(self) -> None:
        a = build_chat_cache_key(_inputs(metadata={"report_b64": "AAAA"}))
        b = build_chat_cache_key(_inputs(metadata={"report_b64": "BBBB"}))
        assert a != b

    def test_workflow_changes_the_key(self) -> None:
        assert build_chat_cache_key(_inputs()) != build_chat_cache_key(_inputs(workflow="clinical"))


class TestVersionDimensions:
    def test_policy_version_is_part_of_the_key(self) -> None:
        assert build_chat_cache_key(_inputs()) != build_chat_cache_key(
            _inputs(policy_version="different")
        )

    def test_model_version_is_part_of_the_key(self) -> None:
        assert build_chat_cache_key(_inputs()) != build_chat_cache_key(
            _inputs(model_version="other-model")
        )

    def test_index_version_is_part_of_the_key(self) -> None:
        assert build_chat_cache_key(_inputs()) != build_chat_cache_key(
            _inputs(index_version="v99")
        )

    def test_versions_default_to_live_system_values(self) -> None:
        """Callers must not have to remember to pass them."""
        from mao.safety.policy import get_policy

        assert _inputs().policy_version == get_policy().policy_version
        assert _inputs().model_version


class TestAttachmentPrivacy:
    def test_raw_attachment_bytes_do_not_appear_in_the_key(self) -> None:
        secret = "SENSITIVE_PATIENT_IMAGE_PAYLOAD"
        key = build_chat_cache_key(_inputs(metadata={"image_b64": secret}))
        assert secret not in key

    def test_raw_query_text_does_not_appear_in_the_key(self) -> None:
        key = build_chat_cache_key(_inputs(query="patient john doe has tau tangles"))
        assert "john doe" not in key
