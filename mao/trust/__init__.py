"""Typed trust classes, the protected input boundary, and the egress gateway.

`01_ARCHITECTURE.md` names seven semantic trust classes and requires that raw
patient content and externally safe text stop being interchangeable `str`
values. This package is where that separation lives.

The public surface is deliberately small. Everything else imports from here:

    RawSensitiveInput        what a caller sent, on any channel
    ProtectedCaseContext     typed clinical facts, private identifiers apart
    SafeEvidenceQuery        minimum facts needed to FIND evidence
    SafeSynthesisContext     minimum facts needed to APPLY evidence to the case
    PublicEvidence           what came back from an approved public source
    ExternalSafePayload      a destination-specific envelope around one of those
    EgressGateway            the one place an external call is authorised
"""
from __future__ import annotations

from mao.trust.classes import (
    ExternalSafePayload,
    ProtectedCaseContext,
    PublicEvidence,
    RawSensitiveInput,
    SafeEvidenceQuery,
    SafeSynthesisContext,
    TrustClass,
)
from mao.trust.egress.gateway import (
    EgressRefused,
    authorise,
    current_protection,
    protected_request,
)
from mao.trust.egress.policy import Destination, EgressPurpose

__all__ = [
    "Destination",
    "EgressPurpose",
    "EgressRefused",
    "ExternalSafePayload",
    "ProtectedCaseContext",
    "PublicEvidence",
    "RawSensitiveInput",
    "SafeEvidenceQuery",
    "SafeSynthesisContext",
    "TrustClass",
    "authorise",
    "current_protection",
    "protected_request",
]
