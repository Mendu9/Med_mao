"""JATS provenance parsing — regression tests for the P2-0 corpus audit defects.

Each test here pins one defect the audit proved in ``mao/data/download_pmc.py``:

D2  ``article-id[@pub-id-type='pmc']`` never matches; the real value is ``pmcid``,
    so every article was skipped and all metadata fell back to defaults.
D3  ``find(".//license/@license-type")`` cannot return an attribute, and JATS
    frequently carries the license in ``ali:license_ref`` instead — so the parsed
    license was *always* the literal ``"open-access"``.
"""
from __future__ import annotations

from pathlib import Path


from mao.corpus.provenance import parse_jats

FIXTURES = Path(__file__).parent.parent / "fixtures" / "corpus"
CCBY_JATS = FIXTURES / "jats_ccby_PMC9102954.xml"


def _wrap(front: str, article_attrs: str = 'article-type="research-article"') -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<pmc-articleset><article {article_attrs}>"
        f"<front>{front}</front><body><p>x</p></body>"
        "</article></pmc-articleset>"
    ).encode("utf-8")


_MINIMAL_FRONT = (
    "<article-meta>"
    '<article-id pub-id-type="pmcid">PMC7777777</article-id>'
    "<title-group><article-title>A Title</article-title></title-group>"
    "</article-meta>"
)


class TestIdentifiers:
    def test_parses_pmcid_from_real_jats(self) -> None:
        """D2: the identifier lives at pub-id-type='pmcid', not 'pmc'."""
        docs = parse_jats(CCBY_JATS.read_bytes())
        assert [d.pmcid for d in docs] == ["PMC9102954"]

    def test_pmcid_is_not_double_prefixed(self) -> None:
        """The XML already carries the PMC prefix; re-adding it corrupts the id."""
        docs = parse_jats(CCBY_JATS.read_bytes())
        assert not docs[0].pmcid.startswith("PMCPMC")

    def test_parses_pmid_and_doi(self) -> None:
        docs = parse_jats(CCBY_JATS.read_bytes())
        assert docs[0].pmid == "35565841"
        assert docs[0].doi == "10.3390/nu14091876"


class TestBibliographicFields:
    def test_parses_title_journal_and_year(self) -> None:
        doc = parse_jats(CCBY_JATS.read_bytes())[0]
        assert doc.journal == "Nutrients"
        assert doc.year == "2022"
        assert doc.title and doc.title != doc.pmcid

    def test_title_is_never_the_bare_pmcid(self) -> None:
        """The legacy corpus stored the PMC id as the title for 92.3% of records."""
        doc = parse_jats(CCBY_JATS.read_bytes())[0]
        assert doc.title != "PMC9102954"

    def test_parses_authors(self) -> None:
        doc = parse_jats(CCBY_JATS.read_bytes())[0]
        assert len(doc.authors) > 0
        assert all(isinstance(a, str) and a.strip() for a in doc.authors)


class TestLicenceExtraction:
    def test_extracts_license_url_from_ali_license_ref(self) -> None:
        """D3: the real license signal is ali:license_ref, not a license-type attr."""
        doc = parse_jats(CCBY_JATS.read_bytes())[0]
        assert doc.license_url == "https://creativecommons.org/licenses/by/4.0/"

    def test_extracts_license_type_attribute_when_present(self) -> None:
        front = _MINIMAL_FRONT.replace(
            "</article-meta>",
            '<permissions><license license-type="open-access">'
            "<license-p>Free to read.</license-p></license></permissions></article-meta>",
        )
        doc = parse_jats(_wrap(front))[0]
        assert doc.license_type == "open-access"

    def test_absent_permissions_yields_unknown_never_open_access(self) -> None:
        """THE regression: a missing license must never become 'open-access'."""
        doc = parse_jats(_wrap(_MINIMAL_FRONT))[0]
        assert doc.license_url == ""
        assert doc.license_type == ""
        assert doc.license_id == "UNKNOWN"

    def test_copyright_statement_is_captured(self) -> None:
        doc = parse_jats(CCBY_JATS.read_bytes())[0]
        assert "2022" in doc.copyright_statement


class TestArticleTypeAndRetraction:
    def test_captures_article_type(self) -> None:
        doc = parse_jats(CCBY_JATS.read_bytes())[0]
        assert doc.article_type == "review-article"

    def test_flags_retraction_article_type(self) -> None:
        doc = parse_jats(_wrap(_MINIMAL_FRONT, 'article-type="retraction"'))[0]
        assert doc.is_retracted is True

    def test_ordinary_article_is_not_flagged_retracted(self) -> None:
        doc = parse_jats(CCBY_JATS.read_bytes())[0]
        assert doc.is_retracted is False


class TestRobustness:
    def test_empty_input_returns_no_documents(self) -> None:
        assert parse_jats(b"") == []

    def test_malformed_xml_returns_no_documents(self) -> None:
        assert parse_jats(b"<article><unclosed>") == []

    def test_article_without_pmcid_is_skipped(self) -> None:
        front = "<article-meta><title-group><article-title>T</article-title></title-group></article-meta>"
        assert parse_jats(_wrap(front)) == []
