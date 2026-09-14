"""
mao/data/download_pmc.py
------------------------
RETIRED. The canonical corpus acquisition path is ``mao/corpus/``.

This module was the original PMC downloader and ChromaDB ingester. The P2-0
corpus audit disproved it: the corpus it produced recorded *no* usable
provenance, and its license field was a broken-XPath fallback rather than a
license. Three compounding defects were proven, each with live evidence:

D1  ``_fetch_metadata_batch`` required biopython, which is not installed, while
    ``_download_fulltext`` used plain ``urllib``. Metadata fetching therefore
    raised on every batch while full text downloaded fine. This is the primary
    root cause of the empty provenance.

D2  ``article-id[@pub-id-type='pmc']`` never matches. Real JATS carries
    ``pub-id-type="pmcid"``, so every article hit the ``if not pmcid: continue``
    guard and was skipped — even with biopython present.

D3  ``find(".//license/@license-type")`` cannot return an attribute:
    ``Element.find()`` returns an Element, and ``@attr`` is not a supported final
    step in ElementPath. Real JATS also frequently carries the license in
    ``ali:license_ref`` with no ``license-type`` attribute at all. The parsed
    license was therefore *always* the literal string ``"open-access"``.

``"open-access"`` is not a license and grants no redistribution right. When the
licenses were actually fetched and parsed, **315 of 900 documents (35%) proved
not redistributable** and one had been **retracted** — none of which was visible
in the data this module produced.

The PMC OA Web Service (``oa.fcgi``) that this module's "OA subset check" relied
on is also retired upstream (HTTP 404), so that check could never have worked.

What to use instead
-------------------
``mao/corpus/`` determines rights from each article's JATS ``<permissions>``
block — the actual license terms, not subset membership — and is fail-closed:
a document is admitted only when a license positively permits redistribution.

    mao/corpus/ncbi.py         dependency-free efetch client
    mao/corpus/provenance.py   JATS parsing
    mao/corpus/licensing.py    normalization + redistribution policy
    mao/corpus/build.py        fail-closed, fully-accounted assembly
    mao/corpus/acquisition.py  deterministic expansion planning

    scripts/fetch_corpus_provenance.py
    scripts/build_corpus_v1.py
    scripts/publish_corpus_artifact.py

Why this file still exists
--------------------------
``mao/data/reingest_all.py`` imports ``_MANIFEST_PATH`` and ``ingest_pmc_papers``
from here. The import surface is preserved so that import keeps working; both
entry points refuse to run and say where to go instead. The original
implementation remains in Git history if it is ever needed as a reference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

#: Where the legacy run recorded its 900 PMCIDs. Still read by tooling that
#: reports on the legacy corpus; the file itself is unchanged.
_MANIFEST_PATH = Path(__file__).parent / "pmc_manifest.json"

_RETIRED = (
    "mao.data.download_pmc is retired: its ingestion recorded no usable provenance "
    "and its license field was always the literal \"open-access\", a broken-XPath "
    "fallback rather than a license (audit defects D1-D3). Re-running it would "
    "rebuild a corpus that cannot be published — 35% of the documents it treated as "
    "open access carry no redistribution right, and one was retracted. "
    "Use mao.corpus instead: scripts/fetch_corpus_provenance.py then "
    "scripts/build_corpus_v1.py. See mao/data/download_pmc.py's docstring for the "
    "full defect record."
)


def download_pmc_papers(
    categories: list[str] | None = None,
    max_per_category: int = 200,
    resume: bool = False,
) -> dict[str, Any]:
    """Retired. Raises :class:`RuntimeError` naming the canonical path."""
    raise RuntimeError(_RETIRED)


def ingest_pmc_papers(
    categories: list[str] | None = None,
    manifest_path: Path | None = None,
) -> int:
    """Retired. Raises :class:`RuntimeError` naming the canonical path."""
    raise RuntimeError(_RETIRED)


def main() -> None:
    """Retired CLI entry point."""
    raise SystemExit(_RETIRED)


if __name__ == "__main__":
    main()
