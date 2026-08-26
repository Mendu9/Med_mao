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

from mao.providers.gateway import model_id_for
from mao.providers.registry import ModelRole
from mao.safety.policy import get_policy

# Bumped by ingestion; overridable so re-indexing invalidates cached answers.
DEFAULT_INDEX_VERSION = "v1"

# Attachment keys whose *content* changes the answer.
_ATTACHMENT_KEYS = ("image_b64", "image_url", "report_b64", "report_path")


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
    index_version: str = field(default_factory=_default_index_version)

    def history_digest(self) -> str:
        """Order-sensitive digest of the conversation so far."""
        return _digest(
            *(f"{t.get('role', '')}:{t.get('content', '')}" for t in self.chat_history)
        )

    def attachment_digest(self) -> str:
        """Digest of attachment content, not just its presence."""
        present = {k: self.metadata.get(k) for k in _ATTACHMENT_KEYS if self.metadata.get(k)}
        if not present:
            return "none"
        return _digest(json.dumps(present, sort_keys=True))


def build_chat_cache_key(inputs: CacheKeyInputs) -> str:
    """Return the Redis key for a chat answer."""
    return "query:" + _digest(
        inputs.user_id,
        inputs.query,
        inputs.workflow,
        inputs.history_digest(),
        inputs.attachment_digest(),
        inputs.policy_version,
        inputs.model_version,
        inputs.index_version,
    )
