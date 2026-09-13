"""NCBI efetch client for document provenance.

All tests inject a fake fetcher: unit tests must never touch the network.
The live integration check lives in ``test_ncbi_live.py``.
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path


from mao.corpus.ncbi import FetchResult, efetch_url, fetch_provenance

FIXTURES = Path(__file__).parent.parent / "fixtures" / "corpus"
CCBY_JATS = FIXTURES / "jats_ccby_PMC9102954.xml"


def _params(url: str) -> dict[str, str]:
    query = urllib.parse.urlparse(url).query
    return {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}


class TestEfetchUrl:
    def test_strips_pmc_prefix_from_ids(self) -> None:
        """The pmc database expects bare numeric ids."""
        assert _params(efetch_url(["PMC9102954", "PMC8873150"]))["id"] == "9102954,8873150"

    def test_requests_full_xml(self) -> None:
        params = _params(efetch_url(["PMC1"]))
        assert params["db"] == "pmc" and params["retmode"] == "xml"

    def test_includes_api_key_when_supplied(self) -> None:
        assert _params(efetch_url(["PMC1"], api_key="SECRET"))["api_key"] == "SECRET"

    def test_omits_api_key_when_absent(self) -> None:
        assert "api_key" not in _params(efetch_url(["PMC1"]))


class TestFetchProvenance:
    def test_returns_documents_keyed_by_pmcid(self) -> None:
        result = fetch_provenance(
            ["PMC9102954"], fetcher=lambda _url: CCBY_JATS.read_bytes(), sleep=0.0
        )
        assert isinstance(result, FetchResult)
        assert "PMC9102954" in result.documents
        assert result.documents["PMC9102954"].doi == "10.3390/nu14091876"

    def test_batches_requests(self) -> None:
        calls: list[str] = []

        def fake(url: str) -> bytes:
            calls.append(url)
            return CCBY_JATS.read_bytes()

        fetch_provenance(
            [f"PMC{n}" for n in range(50)], fetcher=fake, batch_size=20, sleep=0.0
        )
        assert len(calls) == 3

    def test_reports_requested_ids_that_never_came_back(self) -> None:
        result = fetch_provenance(
            ["PMC9102954", "PMC0000001"],
            fetcher=lambda _url: CCBY_JATS.read_bytes(),
            sleep=0.0,
        )
        assert result.missing == ("PMC0000001",)

    def test_a_failing_batch_does_not_abort_the_run(self) -> None:
        """One bad batch must not lose the other 899 articles."""
        def flaky(url: str) -> bytes:
            if "9102954" in url:
                raise OSError("transient network failure")
            return CCBY_JATS.read_bytes()

        result = fetch_provenance(
            ["PMC9102954"], fetcher=flaky, batch_size=1, sleep=0.0
        )
        assert result.documents == {}
        assert result.missing == ("PMC9102954",)

    def test_empty_input_makes_no_requests(self) -> None:
        calls: list[str] = []
        result = fetch_provenance([], fetcher=lambda u: (calls.append(u), b"")[1], sleep=0.0)
        assert calls == []
        assert result.documents == {} and result.missing == ()

    def test_deduplicates_requested_ids(self) -> None:
        calls: list[str] = []

        def fake(url: str) -> bytes:
            calls.append(url)
            return CCBY_JATS.read_bytes()

        fetch_provenance(
            ["PMC9102954", "PMC9102954"], fetcher=fake, batch_size=20, sleep=0.0
        )
        assert len(calls) == 1
