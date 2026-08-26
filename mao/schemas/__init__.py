"""Typed shared schemas.

One place for the objects that cross module boundaries: traces, evidence, and
claims. Business modules import these instead of passing bare dicts around.
"""
from __future__ import annotations

from mao.schemas.evidence import Claim, Evidence, VerificationStatus
from mao.schemas.trace import RetrievalTrace, ToolCallTrace, TraceSchema

__all__ = [
    "Claim",
    "Evidence",
    "VerificationStatus",
    "RetrievalTrace",
    "ToolCallTrace",
    "TraceSchema",
]
