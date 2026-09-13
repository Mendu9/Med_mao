"""Minimal NCBI E-utilities client for PMC document provenance.

Deliberately dependency-free (``urllib`` only). The legacy ingester routed
metadata through ``Bio.Entrez`` while routing full text through ``urllib``; when
biopython was absent the metadata call raised and every article silently fell
back to empty defaults, while the full text downloaded fine. That asymmetry is
the root cause of the empty provenance found by the P2-0 audit.

Note on OA-subset verification: the PMC OA Web Service (``oa.fcgi``) is retired
and returns 404 on both the legacy and current hosts. The per-article JATS
``<permissions>`` block is used instead — it is a stronger signal, because it
states the actual license rather than mere subset membership.
"""
from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from mao.corpus.provenance import DocumentProvenance, parse_jats

logger = logging.getLogger(__name__)

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
USER_AGENT = "mao-corpus/1.0 (https://github.com/Mendu9/Med_mao)"

#: NCBI allows 3 requests/second without an API key, 10 with one.
SLEEP_NO_KEY = 0.35
SLEEP_WITH_KEY = 0.11

DEFAULT_BATCH_SIZE = 20

Fetcher = Callable[[str], bytes]


@dataclass(frozen=True)
class FetchResult:
    """Documents successfully resolved, plus the ids that never came back."""

    documents: dict[str, DocumentProvenance] = field(default_factory=dict)
    missing: tuple[str, ...] = ()

    @property
    def resolved(self) -> int:
        return len(self.documents)


def efetch_url(pmcids: Sequence[str], api_key: str = "", email: str = "") -> str:
    """Build an efetch URL for a batch of PMC ids."""
    bare = [p.upper().replace("PMC", "") for p in pmcids]
    params = {"db": "pmc", "id": ",".join(bare), "rettype": "full", "retmode": "xml"}
    if email:
        params["email"] = email
        params["tool"] = "mao-corpus"
    if api_key:
        params["api_key"] = api_key
    # safe="," keeps the id list in NCBI's documented comma-separated form
    # rather than percent-encoding the separator.
    return f"{EFETCH_URL}?{urllib.parse.urlencode(params, safe=',')}"


def _http_get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return bytes(response.read())


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_provenance(
    pmcids: Sequence[str],
    *,
    fetcher: Fetcher | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    api_key: str = "",
    email: str = "",
    sleep: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FetchResult:
    """Resolve authoritative provenance for ``pmcids`` via NCBI efetch.

    A failing batch is logged and its ids reported as missing; it never aborts
    the run. Callers get an explicit account of what was and was not resolved.
    """
    unique: list[str] = list(dict.fromkeys(p for p in pmcids if p))
    if not unique:
        return FetchResult()

    get = fetcher or _http_get
    pause = SLEEP_WITH_KEY if api_key else SLEEP_NO_KEY if sleep is None else sleep

    documents: dict[str, DocumentProvenance] = {}
    for index, batch in enumerate(_batched(unique, max(1, batch_size))):
        url = efetch_url(batch, api_key=api_key, email=email)
        try:
            raw = get(url)
        except Exception as exc:  # noqa: BLE001 — network/transport of any kind
            logger.warning("efetch batch %d failed (%d ids): %s", index, len(batch), exc)
            raw = b""
        for document in parse_jats(raw):
            documents[document.pmcid] = document
        if progress is not None:
            progress(min((index + 1) * batch_size, len(unique)), len(unique))
        if pause:
            time.sleep(pause)

    missing = tuple(p for p in unique if p not in documents)
    if missing:
        logger.warning("%d of %d PMC ids returned no provenance", len(missing), len(unique))
    return FetchResult(documents=documents, missing=missing)
