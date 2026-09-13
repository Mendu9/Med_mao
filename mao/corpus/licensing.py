"""License normalization and redistribution policy for corpus documents.

The P2-0 audit found the legacy ingester recorded the literal string
``"open-access"`` for every article, produced by a fallback on a broken XPath.
That string is not a license and carries no redistribution right.

This module maps observed JATS license signals onto explicit identifiers and a
single, conservative redistribution decision. The default is always REFUSE: a
document is redistributable only when a recognised license positively says so.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

UNKNOWN = "UNKNOWN"

# Ordered: the first pattern that matches a license URL wins. More specific
# CC variants must precede the plain CC-BY pattern.
_URL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"creativecommons\.org/publicdomain/zero/1\.0", "CC0-1.0"),
    (r"creativecommons\.org/licenses/by-nc-nd/(\d)\.(\d)", "CC-BY-NC-ND-{0}.{1}"),
    (r"creativecommons\.org/licenses/by-nc-sa/(\d)\.(\d)", "CC-BY-NC-SA-{0}.{1}"),
    (r"creativecommons\.org/licenses/by-nc/(\d)\.(\d)", "CC-BY-NC-{0}.{1}"),
    (r"creativecommons\.org/licenses/by-nd/(\d)\.(\d)", "CC-BY-ND-{0}.{1}"),
    (r"creativecommons\.org/licenses/by-sa/(\d)\.(\d)", "CC-BY-SA-{0}.{1}"),
    (r"creativecommons\.org/licenses/by/(\d)\.(\d)", "CC-BY-{0}.{1}"),
)

# JATS ``@content-type`` / ``@license-type`` tokens seen on ali:license_ref.
_TOKEN_MAP: dict[str, str] = {
    "ccbylicense": "CC-BY",
    "ccbynclicense": "CC-BY-NC",
    "ccbyncndlicense": "CC-BY-NC-ND",
    "ccbyncsalicense": "CC-BY-NC-SA",
    "ccbyndlicense": "CC-BY-ND",
    "ccbysalicense": "CC-BY-SA",
    "cc0license": "CC0-1.0",
}

#: License *families* (version suffix stripped) whose terms permit redistributing
#: the full text as a dataset. ND (no-derivatives) variants are excluded: a
#: chunked corpus is a derivative work.
#:
#: NC variants are included because non-commercial licenses permit redistribution
#: and restrict *use*, not distribution. They are recorded separately in the
#: manifest license summary so a downstream consumer can partition them out.
REDISTRIBUTABLE: frozenset[str] = frozenset(
    {"CC0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA"}
)


@dataclass(frozen=True)
class LicenseDecision:
    """Whether a document may be redistributed, and why."""

    license_id: str
    redistributable: bool
    reason: str


def _family(license_id: str) -> str:
    """Strip the version suffix: ``CC-BY-4.0`` -> ``CC-BY``."""
    return re.sub(r"-\d+\.\d+$", "", license_id)


def normalize_license_id(
    license_url: str = "",
    license_type: str = "",
    content_type: str = "",
    license_text: str = "",
) -> str:
    """Map JATS license signals to a normalized identifier.

    Returns :data:`UNKNOWN` when no signal positively identifies a license.
    ``license_type="open-access"`` alone is deliberately *not* an identification:
    it is the exact value the legacy fallback produced.
    """
    for pattern, template in _URL_PATTERNS:
        match = re.search(pattern, license_url or "", re.IGNORECASE)
        if match:
            return template.format(*match.groups()) if match.groups() else template

    token = (content_type or "").strip().lower()
    if token in _TOKEN_MAP:
        return _TOKEN_MAP[token]

    declared = (license_type or "").strip().lower()
    if declared in _TOKEN_MAP:
        return _TOKEN_MAP[declared]

    for pattern, template in _URL_PATTERNS:
        match = re.search(pattern, license_text or "", re.IGNORECASE)
        if match:
            return template.format(*match.groups()) if match.groups() else template

    return UNKNOWN


def decide(license_id: str) -> LicenseDecision:
    """Apply the redistribution policy to a normalized license identifier."""
    if license_id == UNKNOWN or not license_id:
        return LicenseDecision(UNKNOWN, False, "no license identified; refusing by default")

    family = _family(license_id)
    if family in REDISTRIBUTABLE:
        return LicenseDecision(license_id, True, f"{family} permits redistribution")
    if "ND" in family.split("-"):
        return LicenseDecision(
            license_id, False, f"{family} forbids derivatives; a chunked corpus is a derivative"
        )
    return LicenseDecision(license_id, False, f"{family} is not on the redistribution allow-list")
