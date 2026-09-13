"""Parse PMC JATS XML into verified document provenance.

Scoped deliberately to ``front/article-meta``: identifiers also appear inside
``back/ref-list`` citations, and searching the whole article would attribute a
cited paper's DOI to the article itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as SafeET

from mao.corpus.licensing import UNKNOWN, normalize_license_id

logger = logging.getLogger(__name__)

_ALI = "{http://www.niso.org/schemas/ali/1.0/}"
_XLINK = "{http://www.w3.org/1999/xlink}"

_MAX_AUTHORS = 25


@dataclass(frozen=True)
class DocumentProvenance:
    """Verified, document-level provenance for one PMC article."""

    pmcid: str
    pmid: str = ""
    doi: str = ""
    title: str = ""
    journal: str = ""
    year: str = ""
    authors: tuple[str, ...] = field(default_factory=tuple)
    license_url: str = ""
    license_type: str = ""
    license_content_type: str = ""
    license_id: str = UNKNOWN
    copyright_statement: str = ""
    article_type: str = ""
    is_retracted: bool = False

    @property
    def pmc_url(self) -> str:
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{self.pmcid}/"


def _text(node: Element | None) -> str:
    """Flattened text of a node including mixed content, or ``""``."""
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _article_id(meta: Element, id_type: str) -> str:
    return _text(meta.find(f'article-id[@pub-id-type="{id_type}"]'))


def _year(meta: Element) -> str:
    for query in (
        'pub-date[@pub-type="epub"]/year',
        'pub-date[@date-type="pub"]/year',
        'pub-date[@pub-type="ppub"]/year',
        "pub-date/year",
    ):
        value = _text(meta.find(query))
        if value:
            return value
    return ""


def _authors(meta: Element) -> tuple[str, ...]:
    names: list[str] = []
    for contrib in meta.iterfind('.//contrib[@contrib-type="author"]'):
        surname = _text(contrib.find(".//surname"))
        given = _text(contrib.find(".//given-names"))
        full = f"{surname} {given}".strip()
        if full:
            names.append(full)
        if len(names) >= _MAX_AUTHORS:
            break
    return tuple(names)


def _license_signals(meta: Element) -> tuple[str, str, str, str, str]:
    """Return ``(url, license_type, content_type, license_text, copyright)``."""
    permissions = meta.find("permissions")
    if permissions is None:
        return "", "", "", "", ""

    copyright_statement = _text(permissions.find("copyright-statement"))
    lic = permissions.find("license")
    if lic is None:
        return "", "", "", "", copyright_statement

    license_type = (lic.get("license-type") or "").strip()
    url = (lic.get(f"{_XLINK}href") or "").strip()
    content_type = ""

    ref = lic.find(f"{_ALI}license_ref")
    if ref is not None:
        content_type = (ref.get("content-type") or "").strip()
        url = _text(ref) or url

    return url, license_type, content_type, _text(lic), copyright_statement


def _is_retracted(article: Element, meta: Element) -> bool:
    if (article.get("article-type") or "").strip().lower() == "retraction":
        return True
    for related in meta.iterfind("related-article"):
        if (related.get("related-article-type") or "").lower() in {
            "retracted-article",
            "retraction-forward",
        }:
            return True
    return False


def _parse_article(article: Element) -> DocumentProvenance | None:
    meta = article.find("front/article-meta")
    if meta is None:
        return None

    pmcid = _article_id(meta, "pmcid")
    if not pmcid:
        return None
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    url, license_type, content_type, license_text, copyright_statement = _license_signals(meta)

    return DocumentProvenance(
        pmcid=pmcid,
        pmid=_article_id(meta, "pmid"),
        doi=_article_id(meta, "doi"),
        title=_text(meta.find("title-group/article-title")),
        journal=_text(article.find("front/journal-meta/journal-title-group/journal-title"))
        or _text(article.find("front/journal-meta//journal-title")),
        year=_year(meta),
        authors=_authors(meta),
        license_url=url,
        license_type=license_type,
        license_content_type=content_type,
        license_id=normalize_license_id(url, license_type, content_type, license_text),
        copyright_statement=copyright_statement,
        article_type=(article.get("article-type") or "").strip(),
        is_retracted=_is_retracted(article, meta),
    )


def parse_jats(raw_xml: bytes) -> list[DocumentProvenance]:
    """Parse a PMC efetch response into document provenance records.

    Malformed or empty input yields an empty list rather than raising: this
    parses untrusted external data and must never take down an ingest run.
    """
    if not raw_xml or not raw_xml.strip():
        return []
    try:
        root = SafeET.fromstring(raw_xml)
    except Exception as exc:  # noqa: BLE001 — untrusted external input
        logger.warning("JATS parse failed: %s", exc)
        return []

    articles = [root] if root.tag == "article" else list(root.iter("article"))
    documents: list[DocumentProvenance] = []
    for article in articles:
        try:
            parsed = _parse_article(article)
        except Exception as exc:  # noqa: BLE001
            logger.warning("JATS article parse failed: %s", exc)
            continue
        if parsed is not None:
            documents.append(parsed)
    return documents
