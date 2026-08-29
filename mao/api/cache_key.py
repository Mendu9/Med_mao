"""Cache-key construction for the chat endpoints.

A cached answer may only be reused when every input that could change the answer
is identical. The audited key was `md5(f"{user_id}:{query}")`, which ignored
chat history, attachments, and all version dimensions — so re-ingesting the
corpus, switching models, changing the safety policy, or uploading a different
MRI with the same caption all returned a stale answer (P1-4).

Everything is folded through SHA-256, so no raw query text or attachment
payload is recoverable from a key sitting in Redis.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from mao.prompts import registry_version
from mao.providers.gateway import model_id_for
from mao.providers.registry import ModelRole
from mao.safety.policy import get_policy

# Bumped by ingestion; overridable so re-indexing invalidates cached answers.
DEFAULT_INDEX_VERSION = "v1"


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")  # unambiguous separator
    return h.hexdigest()


def _default_index_version() -> str:
    return os.getenv("MAO_INDEX_VERSION", DEFAULT_INDEX_VERSION)


def _default_model_version() -> str:
    return model_id_for(ModelRole.GENERAL_SYNTHESIS)


def _default_prompt_version() -> str:
    return registry_version()


@dataclass(frozen=True)
class CacheKeyInputs:
    """Everything that may change a chat answer."""

    user_id: str
    query: str
    chat_history: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    workflow: str = "evidence_qa"

    policy_version: str = field(default_factory=lambda: get_policy().policy_version)
    model_version: str = field(default_factory=_default_model_version)
    prompt_version: str = field(default_factory=_default_prompt_version)
    index_version: str = field(default_factory=_default_index_version)

    def history_digest(self) -> str:
        """Order-sensitive digest of the conversation so far."""
        return _digest(
            *(f"{t.get('role', '')}:{t.get('content', '')}" for t in self.chat_history)
        )

    def metadata_digest(self) -> str:
        """Digest of ALL request metadata — content, not just presence.

        Every key, not a chosen list. Two rounds of this bug were the same
        mistake at different sizes: first a local four-key tuple that omitted
        audio, so two patients' voice recordings collided on one key; then the
        policy's six-key `ATTACHMENT_KEYS`, which still ignored `patient_id`,
        `encounter_id` and a DICOM study UID — so a caller scoping a request by
        any of those got one cached answer shared across patients.

        A fixed list can only ever enumerate the answer-changing inputs someone
        thought of. Callers send whatever they like, `/chat` consults this cache
        *before* `graph.invoke`, and a cache hit skips the risk gate, the
        council, the judge and the disclaimer. So the default is inverted here:
        everything the caller sent participates, and nothing has to be
        remembered for that to stay true.

        `ATTACHMENT_KEYS` is deliberately no longer consulted here. It answers a
        risk question — which keys mean "patient data is attached" — and this is
        a cache question. Conflating the two is what produced both rounds of the
        bug; the policy keeps owning the risk answer, and this module stops
        borrowing it for something it was never a complete list for.
        """
        present = {k: v for k, v in sorted(self.metadata.items()) if v}
        if not present:
            return "none"
        # `default=str` because a caller can send anything, and a cache key that
        # raises turns an odd metadata value into a 500 on the request path.
        return _digest(json.dumps(present, sort_keys=True, default=str))


def build_chat_cache_key(inputs: CacheKeyInputs) -> str:
    """Return the Redis key for a chat answer."""
    return "query:" + _digest(
        inputs.user_id,
        inputs.query,
        inputs.workflow,
        inputs.history_digest(),
        inputs.metadata_digest(),
        inputs.policy_version,
        inputs.model_version,
        inputs.prompt_version,
        inputs.index_version,
    )
