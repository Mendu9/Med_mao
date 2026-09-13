"""License normalization and the redistribution policy.

The policy is deliberately fail-closed: a document is redistributable only when
a recognised license positively permits it.
"""
from __future__ import annotations

import pytest

from mao.corpus.licensing import UNKNOWN, decide, normalize_license_id


class TestNormalizationFromUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0"),
            ("http://creativecommons.org/licenses/by/3.0/", "CC-BY-3.0"),
            ("https://creativecommons.org/licenses/by-nc/4.0/", "CC-BY-NC-4.0"),
            ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "CC-BY-NC-ND-4.0"),
            ("https://creativecommons.org/licenses/by-sa/4.0/", "CC-BY-SA-4.0"),
            ("https://creativecommons.org/licenses/by-nd/4.0/", "CC-BY-ND-4.0"),
            ("https://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0"),
        ],
    )
    def test_maps_creative_commons_urls(self, url: str, expected: str) -> None:
        assert normalize_license_id(license_url=url) == expected

    def test_by_nc_is_not_mistaken_for_plain_by(self) -> None:
        """The plain by/ pattern must not shadow the more specific variants."""
        assert normalize_license_id(
            license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/"
        ) == "CC-BY-NC-SA-4.0"


class TestNormalizationFromTokens:
    def test_maps_ali_content_type_token(self) -> None:
        assert normalize_license_id(content_type="ccbylicense") == "CC-BY"

    def test_falls_back_to_license_text(self) -> None:
        text = "distributed under the Creative Commons Attribution "
        text += "(CC BY) license (https://creativecommons.org/licenses/by/4.0/)."
        assert normalize_license_id(license_text=text) == "CC-BY-4.0"


class TestTheLegacyDefect:
    def test_open_access_string_alone_is_not_a_license(self) -> None:
        """The exact legacy fallback value must normalize to UNKNOWN."""
        assert normalize_license_id(license_type="open-access") == UNKNOWN

    def test_open_access_string_is_not_redistributable(self) -> None:
        assert decide(normalize_license_id(license_type="open-access")).redistributable is False

    def test_no_signal_at_all_is_unknown(self) -> None:
        assert normalize_license_id() == UNKNOWN


class TestRedistributionPolicy:
    @pytest.mark.parametrize(
        "license_id", ["CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0"]
    )
    def test_permissive_licenses_are_redistributable(self, license_id: str) -> None:
        assert decide(license_id).redistributable is True

    @pytest.mark.parametrize("license_id", ["CC-BY-ND-4.0", "CC-BY-NC-ND-4.0"])
    def test_no_derivatives_licenses_are_refused(self, license_id: str) -> None:
        """A chunked corpus is a derivative work, so ND cannot be redistributed."""
        decision = decide(license_id)
        assert decision.redistributable is False
        assert "derivative" in decision.reason

    def test_unknown_is_refused_by_default(self) -> None:
        decision = decide(UNKNOWN)
        assert decision.redistributable is False
        assert "refusing by default" in decision.reason

    def test_empty_license_id_is_refused(self) -> None:
        assert decide("").redistributable is False

    def test_unrecognised_license_is_refused(self) -> None:
        assert decide("ACME-PROPRIETARY-1.0").redistributable is False

    def test_decision_preserves_the_license_id(self) -> None:
        assert decide("CC-BY-4.0").license_id == "CC-BY-4.0"
