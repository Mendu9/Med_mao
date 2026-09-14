r"""R5 — the domain boost is aimed at a tag almost nothing carries.

`retrieve()` defaults to `domain="alzheimer"` (retriever.py:232) and
`_apply_domain_boost` matches by EXACT EQUALITY against each chunk's stored
`domain` (retriever.py:885). So the x1.2 boost lands only where the two spellings
coincide — and they mostly do not:

    QUERY side   `domain_classifier._VALID_DOMAINS`
                 {"alzheimer", "stroke", "general"}

    CORPUS side  `mao/data/pmc_manifest.json` `category`, and the directory
                 layout under `mao/rag/data/pmc/`
                 {"alzheimers", "stroke", "general_health"}   300 documents each

    INGEST side  `ingest_alzheimers.py:123` hardcodes `"domain": "alzheimer"`
                 on the PDF path; `ingest_pubmed.py:241` writes `"pubmed"`.

The intersection is `stroke` and nothing else. `general` never reaches the
comparison at all — `_apply_domain_boost` early-returns on it — and `alzheimer`
matched only the 42 legacy PDFs, because that is the one ingestion path that
writes the singular spelling.

Corpus V1 EXCLUDES all 42 of those PDFs (no redistribution rights, P2-0 / A2).
So on the corpus the product is being rebuilt around, the DEFAULT domain boost
is now a complete no-op, silently.

## Why this is xfail and not a fix

Two vocabularies describe one concept, nothing checks they agree, and the
mismatch is invisible because the failure mode is "a multiplication that does not
happen". Changing `"alzheimer"` to `"alzheimers"` would be the fifth enumeration
patch in a codebase whose own escalation clause says to stop doing that — it
fixes one spelling and leaves `general`/`general_health` broken, leaves
`pubmed` unreachable from any query label, and leaves nothing detecting the next
drift.

The real remedy is ONE declared domain vocabulary that both sides derive from,
which is `P2-3`'s Source Registry / stable-identifier work, designed against the
contracts `P2-2` produces. It also changes retrieval RANKING, so it must land
where it can be measured rather than as an unmeasured edit to the baseline every
later benchmark is compared against.

`strict=True` on purpose: when the vocabularies are unified this test XPASSes,
which pytest reports as a FAILURE. Whoever fixes R5 is forced to come back here
and delete the marker, so the finding cannot be closed silently or left behind.
"""
from __future__ import annotations

import json
import pathlib

import pytest

_MANIFEST = pathlib.Path("mao/data/pmc_manifest.json")
_PMC_ROOT = pathlib.Path("mao/rag/data/pmc")


def _corpus_domain_tags() -> set[str]:
    """The domain tags the corpus actually carries, from two independent reads.

    Read twice so a single stale file cannot make this test lie: the manifest's
    `category` field and the on-disk directory layout are produced by different
    steps and must agree.
    """
    tags: set[str] = set()
    if _MANIFEST.exists():
        records = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = next(
                (value for value in records.values() if isinstance(value, list)), []
            )
        tags |= {str(record.get("category")) for record in records if record.get("category")}
    if _PMC_ROOT.is_dir():
        tags |= {child.name for child in _PMC_ROOT.iterdir() if child.is_dir()}
    return tags


def test_the_corpus_tags_are_the_ones_this_finding_names() -> None:
    """Non-vacuity, and it must pass. If the corpus stops carrying these tags,
    the xfail below is measuring nothing and this fails first, loudly."""
    assert _corpus_domain_tags() == {"alzheimers", "stroke", "general_health"}


def test_the_default_domain_is_the_one_the_boost_uses() -> None:
    """Also non-vacuity: pins the THREE code locations the finding depends on,
    so a rename elsewhere cannot quietly invalidate the xfail."""
    import inspect

    from mao.agents.domain_classifier import _VALID_DOMAINS
    from mao.rag import retriever

    assert _VALID_DOMAINS == {"alzheimer", "stroke", "general"}
    assert (
        inspect.signature(retriever.retrieve).parameters["domain"].default
        == "alzheimer"
    )


def test_the_boost_still_matches_by_exact_equality() -> None:
    """The third dependency, and the one that makes the xfail below honest.

    The xfail compares two VOCABULARIES. That is only the whole finding while
    matching is exact equality. A fix that unified the vocabularies by changing
    the MATCHING instead — normalising, aliasing, stemming, prefix-matching —
    would leave the xfail still xfailing, so nobody would be forced back to
    delete the marker, which is precisely what `strict=True` exists to prevent.

    So the pin asserts the mechanism, not just the spellings: `alzheimer` must
    fail to boost a chunk tagged `alzheimers`.
    """
    from mao.rag.retriever import _apply_domain_boost

    chunks = [
        {"chunk_id": "a", "domain": "alzheimers", "score": 0.50},
        {"chunk_id": "b", "domain": "stroke", "score": 0.50},
    ]
    boosted = _apply_domain_boost(chunks, domain="alzheimer", factor=1.2)

    by_id = {chunk["chunk_id"]: chunk["score"] for chunk in boosted}
    assert by_id["a"] == 0.50, (
        "'alzheimer' boosted an 'alzheimers' chunk — matching is no longer exact "
        "equality, so the vocabulary comparison below is no longer the whole "
        "finding. Re-derive R5 before trusting the xfail."
    )
    assert by_id["b"] == 0.50, "unrelated domain must be untouched"

    # The control: the one label that DOES intersect the corpus still boosts,
    # so this test cannot pass by the boost being broken outright.
    stroke = _apply_domain_boost(
        [{"chunk_id": "c", "domain": "stroke", "score": 0.50}],
        domain="stroke",
        factor=1.2,
    )
    assert stroke[0]["score"] == pytest.approx(0.60)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "R5 — query-side domain labels and corpus-side domain tags are two "
        "vocabularies with no shared definition. Unify in P2-3 against the "
        "P2-2 contracts; do not patch one spelling."
    ),
)
def test_every_query_domain_label_can_boost_something() -> None:
    """The invariant R5 violates: a domain the classifier can emit must be
    capable of boosting a chunk the corpus actually contains. Otherwise the
    feature is inert for that domain and nothing says so."""
    from mao.agents.domain_classifier import _VALID_DOMAINS

    corpus_tags = _corpus_domain_tags()

    # `general` is excluded from the comparison by `_apply_domain_boost`'s own
    # early return, so it is held to the weaker obligation of being a DECLARED
    # no-op rather than an accidental one — it is still counted here because
    # `general_health` exists and a reader would reasonably expect them to meet.
    inert = sorted(label for label in _VALID_DOMAINS if label not in corpus_tags)

    assert inert == [], (
        f"domain label(s) {inert} can never boost a chunk: the corpus carries "
        f"{sorted(corpus_tags)}. The x1.2 boost is inert on the default path."
    )
