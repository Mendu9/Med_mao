---
license: other
license_name: mixed-creative-commons
language:
  - en
task_categories:
  - text-retrieval
tags:
  - biomedical
  - pubmed-central
  - evidence-retrieval
size_categories:
  - n<1K
---

# MAO Corpus V1 — source documents

A license-verified, document-level biomedical corpus assembled from PubMed
Central full text. **584 documents**, one row per document, every row carrying
its own provenance and a positively identified redistribution license.

This dataset is the *source* artifact. Chunking, embeddings and sparse indexes
are derived artifacts and are published separately — a corpus and an index built
from it are never the same artifact.

## Why this exists

It replaces a 161 MB JSON blob that labelled **every** document `"open-access"`.
That string was the fallback value of a broken XPath, not a license claim. When
the licenses were actually fetched and parsed from each article's JATS
`<permissions>` block, 315 of 900 documents (35%) turned out **not** to be
redistributable, and one had been **retracted**. None of that was detectable in
the original data.

## Contents

| File | Rows | Description |
|---|---|---|
| `documents.parquet` | 584 | One row per admitted document: full text plus verified provenance |
| `corpus_manifest.json` | — | Counts, license summary, per-file SHA-256 |
| `exclusions.json` | 316 | Every rejected candidate with an explicit reason |
| `provenance.json` | 900 | Resolved provenance for all candidates, admitted or not |

### `documents.parquet` schema

| Column | Type | Notes |
|---|---|---|
| `pmcid` | string | PMC identifier; unique, 0 duplicates |
| `title`, `journal`, `year`, `doi`, `pmid` | string | Fetched from NCBI efetch, not inferred |
| `authors` | list\<string\> | |
| `article_type` | string | JATS article type |
| `license_id` | string | Normalized SPDX-style identifier; never `UNKNOWN` |
| `license_url` | string | The license URL as declared by the publisher |
| `copyright_statement` | string | Verbatim from the article |
| `pmc_url` | string | Canonical source URL |
| `text` | string | Full document text |
| `sha256` | string | Content hash of `text` |
| `word_count`, `char_count` | int64 | |

## Licensing — read this before redistributing

Licenses are **per document**, not per dataset. The `license_id` column is
authoritative for each row.

| License | Documents |
|---|---|
| CC-BY-4.0 | 460 |
| CC-BY-NC-4.0 | 102 |
| CC-BY-3.0 | 7 |
| CC-BY-NC-SA-4.0 | 7 |
| CC-BY-NC-3.0 | 6 |
| CC-BY-2.0 | 1 |
| CC-BY-NC-SA-1.0 | 1 |
| **UNKNOWN** | **0** |

**116 documents carry a NonCommercial (NC) clause.** They are recorded
separately so a consumer can partition them out; filter on `license_id` before
any commercial use. All licenses require attribution — the `authors`, `title`,
`journal`, `doi` and `license_url` columns carry what is needed to give it.

### Admission policy

Fail-closed. A document is admitted only when a license is *positively*
identified as permitting redistribution:

- **Allowed:** CC0, CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA.
- **Refused:** every **ND** (no-derivatives) variant — a chunked corpus is a
  derivative work — and anything with no identified license.
- **Refused:** the bare string `"open-access"`, which normalizes to `UNKNOWN`.
  That was the legacy fallback and is not a license.
- **Refused:** retracted articles — not admissible as clinical evidence.

Every candidate is either admitted or recorded in `exclusions.json` with a
reason. Nothing is dropped silently.

## Intended use and limitations

Built for retrieval research and evidence grounding. It is **not** a clinical
decision tool and carries no medical authority.

Known limitations, measured rather than assumed:

- **Document boundaries are inherited from the legacy ingestion and are
  imperfect.** The largest admitted document is 120,783 words and is almost
  certainly a whole supplement ingested as one article. Re-derivation is planned.
- **Topically narrow.** Selection targeted stroke, Alzheimer's disease and
  general health; it is not a representative sample of biomedical literature.
- **Retraction status reflects the fetch date** recorded in `corpus_manifest.json`.
  A paper retracted after that date will not be flagged. Re-check before relying
  on any document as current evidence.
- **Year range 2015–2026**, so older foundational literature is absent.

## Provenance

Fetched from NCBI E-utilities (`efetch`, `db=pmc`) against a recorded PMCID list.
License and retraction status come from each article's JATS `<permissions>` and
article-type metadata — actual license terms, not subset membership. The PMC OA
Web Service (`oa.fcgi`) is retired (HTTP 404) and is deliberately not used.

## Integrity

Every file is pinned by SHA-256 and byte size in `corpus_manifest.json`, and the
consuming code refuses to load an artifact whose bytes disagree with its pin.
Address this dataset by an immutable commit revision, never by a branch.
