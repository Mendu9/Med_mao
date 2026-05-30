# Agent 1 — RAG Pipeline Specialist: Comprehensive Test Report

**Date:** 2026-05-11
**Project:** MAO Biomedical GraphRAG
**Agent Role:** RAG Pipeline Specialist
**Scope:** Chunking, dense vector search, BM25 sparse retrieval, RRF merge, reranker, and retrieval metrics (Precision@k, Recall@k, Hit Rate)

---

## 1. Executive Summary

The MAO RAG pipeline implements a well-architected 9-step GraphRAG retrieval system. Static code analysis and manual metric computation against the golden dataset reveal **2 CRITICAL bugs**, **2 HIGH issues**, and **4 MEDIUM issues**. The core chunker and BM25 components are functionally sound. The reranker has a hard import failure at module level, and the golden-dataset regression test contains a patching error that renders all 20 parametrized test cases non-functional.

---

## 2. Test Results Table

| # | Test Name | Component | Result | Metric / Finding |
|---|-----------|-----------|--------|-----------------|
| T-01 | `test_rag_keywords_present[alz_001]` | Keyword coverage | PASS | donepezil, memantine, palliative — all found |
| T-02 | `test_rag_keywords_present[alz_002]` | Keyword coverage | PASS | amyloid, plaques, tau — all found |
| T-03 | `test_rag_keywords_present[alz_003]` | Keyword coverage | PASS | exercise, diet, cognitive — all found |
| T-04 | `test_rag_keywords_present[alz_004]` | Keyword coverage | PASS | apoe4, risk, allele — all found |
| T-05 | `test_rag_keywords_present[alz_005]` | Keyword coverage | PASS | mmse, mri, biomarkers — all found |
| T-06 | `test_rag_keywords_present[alz_006]` | Keyword coverage | PASS | acetylcholine, acetylcholinesterase, donepezil — all found |
| T-07 | `test_rag_keywords_present[alz_007]` | Keyword coverage | PASS | early-onset, late-onset, genetic — all found |
| T-08 | `test_rag_keywords_present[alz_008]` | Keyword coverage | PASS | microglia, inflammation, cytokines — all found |
| T-09 | `test_rag_keywords_present[str_001]` | Keyword coverage | PASS | fast, face, arm — all found |
| T-10 | `test_rag_keywords_present[str_002]` | Keyword coverage | PASS | tpa, thrombolytic, window — all found |
| T-11 | `test_rag_keywords_present[str_003]` | Keyword coverage | PASS | hemorrhage, hypertension, aneurysm — all found |
| T-12 | `test_rag_keywords_present[str_004]` | Keyword coverage | PASS | physiotherapy, speech, recovery — all found |
| T-13 | `test_rag_keywords_present[gen_001]` | Keyword coverage | PASS | algorithm, data, predictions — all found |
| T-14 | `test_rag_keywords_present[gen_002]` | Keyword coverage | PASS | neurons, layers, weights — all found |
| T-15 | `test_rag_keywords_present[gen_003]` | Keyword coverage | PASS | entities, nodes, graph — all found |
| T-16 | `test_rag_keywords_present[gen_004]` | Keyword coverage | PASS | tokens, embedding, transformer — all found |
| T-17 | `test_golden_query[*]` (20 cases) | Live retrieval mock | FAIL | Wrong patch target `_embed` (DNE); `ids` key missing from mock; dataclass accessed as dict |
| T-18 | BM25 index build (static analysis) | BM25 | PASS | Tokenization correct; optional dep guard works |
| T-19 | BM25 search graceful degradation | BM25 | PASS | Returns `[]` when index is None |
| T-20 | `adaptive_biomedical_chunk` bounds | Chunker | PASS | min/max_words enforced in adaptive path |
| T-21 | `section_aware_split` sections | Chunker | PASS | Regex captures Abstract/Methods/Results correctly |
| T-22 | `rerank()` sort order | Reranker | PASS | Results sorted descending, top_k respected |
| T-23 | `rerank()` unavailability fallback | Reranker | PASS | Score passthrough works when `_reranker` is None |
| T-24 | `FlagEmbedding` module import | Reranker | FAIL | Hard top-level import — no try/except guard (BUG-001) |
| T-25 | `_reciprocal_rank_fusion` math | Retriever | PASS | RRF scores correct; k=60 standard constant |
| T-26 | `_vector_search` score formula | Retriever | FAIL | `1.0 - dist` gives negative scores for L2 dist > 1.0 (BUG-004) |
| T-27 | `QueryCache.set` with RankedChunk | Cache | FAIL | `json.dumps(RankedChunk)` raises TypeError; Redis cache silently broken (BUG-003) |
| T-28 | `_fixed_chunk` min_words floor | Chunker | FAIL | Last chunk may be below min_words (BUG-005) |

---

## 3. Manual Retrieval Metric Computation

### 3.1 Methodology

Source data: `tests/eval/golden/dataset.json` (20 entries) and `tests/eval/fixtures/chunks.json`.

For each entry with non-empty `expected_keywords` and fixture chunks (16 qualifying queries):

- **Relevant chunk** = any chunk whose `.text` (lowercased) contains at least one expected keyword.
- **Top-5** = all fixture chunks for that query ID (fixture provides at most 3 per query, which bounds the denominator).
- **Precision@5** = relevant_found / total_fixture_chunks
- **Recall@5** = relevant_found / total_relevant_in_fixture (same denominator — fixture is ground truth)
- **Hit Rate** = 1 if any relevant chunk exists, else 0

### 3.2 Per-Query Metrics

| Query ID | Keywords | Fixture Chunks | Relevant | Precision@5 | Recall@5 | Hit Rate |
|----------|----------|:-:|:-:|:-:|:-:|:-:|
| alz_001 | donepezil, memantine, palliative | 3 | 3 | 1.000 | 1.000 | 1 |
| alz_002 | amyloid, plaques, tau | 2 | 2 | 1.000 | 1.000 | 1 |
| alz_003 | exercise, diet, cognitive | 3 | 3 | 1.000 | 1.000 | 1 |
| alz_004 | apoe4, risk, allele | 2 | 2 | 1.000 | 1.000 | 1 |
| alz_005 | mmse, mri, biomarkers | 3 | 3 | 1.000 | 1.000 | 1 |
| alz_006 | acetylcholine, acetylcholinesterase, donepezil | 2 | 2 | 1.000 | 1.000 | 1 |
| alz_007 | early-onset, late-onset, genetic | 2 | 2 | 1.000 | 1.000 | 1 |
| alz_008 | microglia, inflammation, cytokines | 2 | 2 | 1.000 | 1.000 | 1 |
| str_001 | fast, face, arm | 2 | 2 | 1.000 | 1.000 | 1 |
| str_002 | tpa, thrombolytic, window | 2 | 2 | 1.000 | 1.000 | 1 |
| str_003 | hemorrhage, hypertension, aneurysm | 2 | 2 | 1.000 | 1.000 | 1 |
| str_004 | physiotherapy, speech, recovery | 2 | 2 | 1.000 | 1.000 | 1 |
| gen_001 | algorithm, data, predictions | 2 | 2 | 1.000 | 1.000 | 1 |
| gen_002 | neurons, layers, weights | 2 | 2 | 1.000 | 1.000 | 1 |
| gen_003 | entities, nodes, graph | 2 | 2 | 1.000 | 1.000 | 1 |
| gen_004 | tokens, embedding, transformer | 2 | 2 | 1.000 | 1.000 | 1 |

Excluded from metrics (no keywords or no fixture chunks):
- `unsafe_001`, `unsafe_002`: correctly empty fixtures — no retrieval for unsafe queries
- `edge_001`: empty query — correctly empty
- `edge_002`: no expected keywords field populated

### 3.3 Aggregate Metrics

| Metric | Value | Queries Evaluated |
|--------|-------|:-:|
| **Precision@5 (mean)** | **1.000** | 16 |
| **Recall@5 (mean)** | **1.000** | 16 |
| **Hit Rate (mean)** | **1.000** | 16 |

**Interpretation:** The fixture dataset is hand-curated so every chunk is relevant to its query. These scores validate fixture correctness, not the live retriever. The `test_golden_dataset.py` tests that exercise the live retriever via mock are broken (see BUG-002) and must be fixed before pipeline regression testing is meaningful.

---

## 4. Component-Level Analysis

### 4.1 BM25 Sparse Retrieval (`mao/rag/retriever.py` lines 318–389)

| Check | Result | Notes |
|-------|--------|-------|
| Tokenization (`text.lower().split()`) | PASS | Standard whitespace tokenization |
| Optional dependency guard (`try: from rank_bm25`) | PASS | No-op if not installed |
| Graceful return `[]` when index is None | PASS | Correct degradation |
| Score filter `scores[idx] > 0` | PASS | Eliminates zero-overlap noise |
| Top-N selection via `np.argsort(scores)[::-1][:n]` | PASS | Correct descending sort |
| Corpus / chunk_id alignment | PASS | Parallel lists maintained correctly |

**Result: PASS** — BM25 implementation is correct and degrades gracefully.

---

### 4.2 Chunker (`mao/rag/chunker.py`)

| Check | Result | Notes |
|-------|--------|-------|
| `adaptive_biomedical_chunk` respects `max_words` | PASS | `current_words + word_count > max_words` triggers split |
| `adaptive_biomedical_chunk` respects `min_words` | PASS | `current_words >= min_words` guards split |
| Empty input returns `[]` | PASS | Line 54 guard |
| Single-sentence input returns `[text]` | PASS | Line 73–74 guard |
| `section_aware_split` preamble handling | PASS | Text before first section stored as "preamble" |
| `section_aware_split` no-section fallback | PASS | Returns `{"body": text}` |
| `_fixed_chunk` fallback called when model unavailable | PASS | logger.warning issued |
| `_fixed_chunk` min_words enforcement | FAIL | Last chunk can be stub (see BUG-005) |
| Chunk ID uniqueness | MEDIUM | MD5 hash; collision risk at scale (see BUG-006) |

**Result: PASS with caveats** — adaptive path is correct; fixed fallback has trailing stub issue.

---

### 4.3 Reranker (`mao/rag/reranker.py`)

| Check | Result | Notes |
|-------|--------|-------|
| Module-level `FlagEmbedding` import | FAIL | Hard import, no try/except — CRITICAL |
| `_get_reranker()` `_reranker_failed` guard | UNREACHABLE | Import fails before this code runs |
| `rerank()` fallback when reranker None | PASS (code path) | Would work if import were guarded |
| Sort order descending by score | PASS | `reverse=True` in sorted() |
| `top_k` respected | PASS | `ranked[:k]` slice |
| Inference failure fallback | PASS | Returns original order with score=0.0 |
| `RankedChunk.metadata` type | MEDIUM | No default value; callers must always pass metadata |

**Result: FAIL** — module cannot be imported without FlagEmbedding installed.

---

### 4.4 RRF Merge (`mao/rag/retriever.py` lines 392–424)

Manual trace with 2 lists:
```
List A: [chunk_X (rank 0), chunk_Y (rank 1)]
List B: [chunk_X (rank 0), chunk_Z (rank 1)]

chunk_X: 1/(60+1) + 1/(60+1) = 0.032787
chunk_Y: 1/(60+2)             = 0.016129
chunk_Z: 1/(60+2)             = 0.016129

Final order: chunk_X > chunk_Y = chunk_Z  (correct)
```

| Check | Result |
|-------|--------|
| RRF formula `1/(k + rank + 1)` | PASS |
| Deduplication by chunk_id | PASS |
| Fallback key to `text[:50]` | PASS |
| k=60 constant (Cormack 2009) | PASS |
| Output sorted descending | PASS |

**Result: PASS**

---

### 4.5 Vector Search Score (`mao/rag/retriever.py` line 222)

```python
"score": 1.0 - dist,   # dist = ChromaDB L2 distance
```

- ChromaDB default distance metric: **L2** (Euclidean), range [0, +inf)
- For unit-norm vectors: L2 range [0, 2], so `1.0 - dist` range [-1.0, 1.0]
- For non-unit-norm vectors: dist can exceed 2.0, score becomes < -1.0

**Result: FAIL** — score formula assumes cosine distance or normalized vectors; produces negative scores otherwise (BUG-004).

---

## 5. Bug Inventory

### BUG-001 — CRITICAL: Hard top-level FlagEmbedding import

**File:** `mao/rag/reranker.py` line 25
**Root cause:** `from FlagEmbedding import FlagReranker` is a module-level statement with no `try/except`. If `FlagEmbedding` is absent (fresh venv, CI runner, Docker build), the entire module raises `ImportError` at import time. This takes down `mao.rag.retriever` which does `from mao.rag.reranker import RankedChunk, rerank` at line 53.

The `_reranker_failed` guard (line 44) and the `try/except` in `_get_reranker()` (lines 53–59) are both **unreachable** because the crash happens before any function is called.

**Exact fix:**
```python
# mao/rag/reranker.py — replace line 25

# BEFORE:
from FlagEmbedding import FlagReranker  # pip install FlagEmbedding

# AFTER:
try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None  # type: ignore[assignment,misc]
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "FlagEmbedding not installed — reranker will use score passthrough. "
        "pip install FlagEmbedding"
    )
```

Also update the type annotation on line 41:
```python
# BEFORE:
_reranker: FlagReranker | None = None
# AFTER:
_reranker: "FlagReranker | None" = None
```

---

### BUG-002 — CRITICAL: Wrong patch target and wrong result access in test_golden_dataset.py

**File:** `tests/eval/test_golden_dataset.py` lines 46–65
**Root cause (three sub-issues):**

1. `patch("mao.rag.retriever._embed")` — `_embed` does not exist in `retriever.py`. The actual function is `embed_query` imported from `mao.rag.embedder`. All 20 parametrized tests apply a no-op patch and the real `embed_query` fires against a non-running embedding server.

2. `mock_col.query.return_value` is missing the `"ids"` key. `_vector_search` (retriever.py line 214) unpacks `results.get("ids", [[]])[0]`, which returns `[]`, so the `zip(docs, metas, ids, distances)` produces zero iterations and `retrieve()` returns an empty list even with the mock active.

3. Line 64: `r.get("text", "")` — `retrieve()` returns `list[RankedChunk]` (dataclasses). `RankedChunk` has no `.get()` method. This raises `AttributeError` even if the mock were corrected.

**Exact fix:**
```python
# tests/eval/test_golden_dataset.py — replace lines 45-66

@pytest.mark.slow
@pytest.mark.parametrize("query,keywords", GOLDEN_DATASET)
def test_golden_query(query: str, keywords: list[str]) -> None:
    """Retrieved chunks for each golden query must contain all required keywords."""
    fake_text = " ".join(keywords) + " " + query

    with patch("mao.rag.embedder.embed_query", return_value=[0.0] * 384), \
         patch("mao.rag.retriever._get_collection") as mock_get_col, \
         patch("mao.rag.graph_builder.load_graph") as mock_graph, \
         patch("mao.rag.reranker._get_reranker", return_value=None):

        mock_col = mock_get_col.return_value
        mock_col.count.return_value = 1
        mock_col.query.return_value = {
            "documents": [[fake_text]],
            "metadatas": [[{"source": "test.pdf", "domain": "alzheimer", "entities": ""}]],
            "ids": [["chunk_test_001"]],
            "distances": [[0.1]],
        }
        import networkx as nx
        mock_graph.return_value = nx.MultiDiGraph()

        from mao.rag.retriever import retrieve
        results = retrieve(query=query, domain="alzheimer", top_k=3)

    assert results, f"No results for: {query}"
    # results is list[RankedChunk] — access .text attribute, not .get()
    combined = " ".join(r.text for r in results).lower()
    for kw in keywords:
        assert kw.lower() in combined, f"Keyword '{kw}' not found for query: {query}"
```

---

### BUG-003 — HIGH: QueryCache.set fails to serialize RankedChunk to Redis

**File:** `mao/core/query_cache.py` line 52
**Root cause:** `json.dumps(value)` where `value` is `list[RankedChunk]`. `RankedChunk` is a `@dataclass` instance, which is not JSON-serializable by default. `json.dumps` raises `TypeError: Object of type RankedChunk is not JSON serializable`. The exception is caught silently (line 53 logs at DEBUG, not WARNING), and execution falls through to the in-memory store. The Redis cache is therefore **always bypassed** for retrieve() results in production.

**Exact fix:**
```python
# mao/core/query_cache.py — replace set() method

def set(self, key: str, value: Any) -> None:
    if self._redis is not None:
        try:
            import dataclasses

            def _default(obj: Any) -> Any:
                if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                    return dataclasses.asdict(obj)
                raise TypeError(f"Type {type(obj)} not JSON serializable")

            self._redis.set(
                f"qcache:{key}",
                json.dumps(value, default=_default),
                ex=self._ttl,
            )
            return
        except Exception as exc:
            logger.debug("Redis cache SET failed, using in-memory: %s", exc)
    self._store[key] = (value, time.monotonic())
```

And the corresponding `get()` method needs a deserializer that reconstructs `RankedChunk` from dict:
```python
# In get(), after json.loads(raw):
from mao.rag.reranker import RankedChunk
raw_list = json.loads(raw)
if isinstance(raw_list, list) and raw_list and isinstance(raw_list[0], dict) and "text" in raw_list[0]:
    return [RankedChunk(**item) for item in raw_list]
return raw_list
```

---

### BUG-004 — HIGH: _vector_search negative scores for L2 distance > 1.0

**File:** `mao/rag/retriever.py` line 222
**Root cause:** `"score": 1.0 - dist` assumes ChromaDB is configured for cosine distance (range [0, 1]). ChromaDB's default distance metric is L2. For unit-norm embeddings (as produced by `all-MiniLM-L6-v2`), L2 distance range is [0, 2], making the minimum score -1.0. For non-unit-norm embeddings, distances can exceed 2.0 producing scores below -1.0. Downstream, negative scores interfere with RRF fusion score comparisons and any score thresholding logic.

**Exact fix (Option A — normalize score):**
```python
# mao/rag/retriever.py line 222
# BEFORE:
"score": 1.0 - dist,
# AFTER: (maps L2 dist [0,2] for unit vectors to score [0,1])
"score": max(0.0, 1.0 - dist / 2.0),
```

**Exact fix (Option B — configure collection for cosine, preferred):**
```python
# mao/rag/retriever.py — _get_collection(), line 74
_chroma_collection = _chroma_client.get_or_create_collection(
    cfg.chroma_collection,
    metadata={"hnsw:space": "cosine"},  # distance range [0,1] for unit vectors
)
# Then line 222 can remain: "score": 1.0 - dist
```

---

### BUG-005 — MEDIUM: _fixed_chunk ignores min_words for trailing chunk

**File:** `mao/rag/chunker.py` lines 187–195
**Root cause:** `_fixed_chunk` (the fallback path when sentence-transformers is unavailable) splits text into `max_words`-sized windows without enforcing `min_words`. The last window will often be smaller than `min_words`. For example, a 410-word text with `max_words=400` yields a second chunk of 10 words — far below the 60-word minimum.

**Exact fix:**
```python
# mao/rag/chunker.py — replace _fixed_chunk

def _fixed_chunk(text: str, max_words: int = 400, min_words: int = 0) -> list[str]:
    """Fallback: split into fixed-size word chunks.

    Merges trailing stub chunks (< min_words) into the preceding chunk
    rather than emitting them as standalone fragments.
    """
    words = text.split()
    raw_chunks = [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
        if words[i : i + max_words]
    ]
    if not raw_chunks:
        return []

    # Merge trailing stub into previous chunk if below min_words
    if min_words > 0 and len(raw_chunks) > 1:
        last = raw_chunks[-1]
        if len(last.split()) < min_words:
            raw_chunks[-2] = raw_chunks[-2] + " " + last
            raw_chunks = raw_chunks[:-1]

    return [c for c in raw_chunks if c.strip()]
```

Also update the call sites to pass `min_words`:
```python
# adaptive_biomedical_chunk line 60:
return _fixed_chunk(text, max_words, min_words)

# chunk_document line 164 — already passes through adaptive_biomedical_chunk which passes to _fixed_chunk
```

---

### BUG-006 — MEDIUM: MD5 used for chunk_id — collision risk at scale

**File:** `mao/rag/chunker.py` line 171
**Root cause:** `hashlib.md5(f"{doc_id}:{chunk_index}:{chunk_text[:50]}".encode()).hexdigest()` uses only the first 50 characters of chunk text. For large corpora (>100K chunks), MD5's 128-bit space presents a birthday collision probability of ~0.1% at 1M chunks. Combined with the truncated input, semantically similar documents with common prefixes increase practical collision risk.

**Exact fix:**
```python
# mao/rag/chunker.py line 171
# BEFORE:
chunk_id = hashlib.md5(f"{doc_id}:{chunk_index}:{chunk_text[:50]}".encode()).hexdigest()
# AFTER:
chunk_id = hashlib.sha256(
    f"{doc_id}:{chunk_index}:{chunk_text[:100]}".encode()
).hexdigest()[:32]
```

---

### BUG-007 — MEDIUM: Synonym map rebuilt on every retrieve() call

**File:** `mao/rag/retriever.py` — `_expand_query()` function (line 427–479)
**Root cause:** `_expand_query` calls `load_graph()` and then `build_synonym_map(G)` on every invocation. `build_synonym_map` iterates all graph nodes to build a synonym dict. For a graph with tens of thousands of entities this is O(N) per query with no caching.

**Exact fix:**
```python
# mao/rag/retriever.py — add module-level cache after BM25 state block

_synonym_map_cache: "dict[str, str] | None" = None

def _expand_query(query: str) -> str:
    global _synonym_map_cache
    try:
        from mao.rag.ontology_loader import build_synonym_map
        G = load_graph()
        if G.number_of_nodes() == 0:
            return query
        if _synonym_map_cache is None:
            _synonym_map_cache = build_synonym_map(G)
        synonym_map = _synonym_map_cache
        # ... rest of function unchanged
```

---

### BUG-008 — LOW: logger referenced before definition in retriever.py

**File:** `mao/rag/retriever.py` lines 70–78
**Root cause:** `_get_collection()` is defined at lines 70–76 and references `logger` at line 75. `logger = logging.getLogger(__name__)` is assigned at line 78. Python resolves names in function bodies at call time (not definition time), so this works at runtime — but it violates the convention of defining module-level names before the functions that use them and will be flagged by linters.

**Exact fix:** Move `logger = logging.getLogger(__name__)` from line 78 to before `_get_collection()` (before line 70).

---

## 6. Bug Summary Table

| Bug ID | Severity | File | Line | Issue |
|--------|----------|------|------|-------|
| BUG-001 | CRITICAL | `mao/rag/reranker.py` | 25 | Hard top-level `FlagEmbedding` import — no try/except guard |
| BUG-002 | CRITICAL | `tests/eval/test_golden_dataset.py` | 46, 64 | Wrong patch target (`_embed` DNE); missing `ids` key; `.get()` on dataclass |
| BUG-003 | HIGH | `mao/core/query_cache.py` | 52 | `json.dumps(RankedChunk)` TypeError; Redis cache silently disabled |
| BUG-004 | HIGH | `mao/rag/retriever.py` | 222 | `1.0 - dist` negative for L2 distance > 1.0 |
| BUG-005 | MEDIUM | `mao/rag/chunker.py` | 187–195 | `_fixed_chunk` ignores `min_words`; trailing stub chunks |
| BUG-006 | MEDIUM | `mao/rag/chunker.py` | 171 | MD5 chunk ID — collision risk at scale |
| BUG-007 | MEDIUM | `mao/rag/retriever.py` | 451 | Synonym map rebuilt O(N) per query; no cache |
| BUG-008 | LOW | `mao/rag/retriever.py` | 75 | `logger` referenced before definition (runtime-safe; linting issue) |

---

## 7. Fixture Quality Assessment

| Aspect | Finding |
|--------|---------|
| Golden dataset size | 20 entries: 8 Alzheimer, 4 stroke, 4 general, 2 unsafe, 2 edge |
| Keyword coverage in fixtures | 100% — all expected keywords found in corresponding chunks |
| Unsafe entries | Correctly have empty chunk lists — no retrieval for harmful queries |
| Edge cases | Empty query and off-topic query correctly produce empty/irrelevant chunks |
| Fixture entries vs golden IDs | All 20 golden IDs have corresponding entries in `chunks.json` |
| Chunk score range | 0.83–0.93 (realistic cosine similarity values) |
| Missing `entities` metadata | No chunk in `chunks.json` has an `entities` field — the metadata filter path in `_entity_search` (retriever.py line 262) will always fall through to vector fallback |

**Note:** The absence of `entities` metadata in fixture chunks means the primary `_entity_search` code path (ChromaDB `$contains` filter) is never exercised by any existing test. This is a test coverage gap, not a bug.

---

## 8. Test Coverage Gaps

| Gap | Impact | Recommendation |
|-----|--------|----------------|
| No test for `_build_bm25_index` + `_bm25_search` roundtrip | BM25 path untested end-to-end | Add `tests/eval/test_bm25.py` |
| No test for `_reciprocal_rank_fusion` | RRF merge untested | Add unit test with controlled ranked lists |
| No test for `_entity_search` metadata filter path | Primary entity search path untested | Add test with mock collection including `entities` metadata |
| No test for `QueryCache` with `RankedChunk` values | Cache serialization bug not caught | Add unit test serializing actual dataclass values |
| No test for `adaptive_biomedical_chunk` boundary detection | Semantic chunking logic untested | Add test with mocked sentence-transformer model |
| `test_golden_dataset.py` uses `@pytest.mark.slow` | Excluded from standard CI | Confirm slow tests run in nightly pipeline |

---

## 9. Recommendations (Priority Order)

1. **[CRITICAL]** Fix `FlagEmbedding` import guard (`reranker.py` line 25) — prevents the entire pipeline from loading in clean environments.
2. **[CRITICAL]** Fix `test_golden_dataset.py` patch targets — all 20 live-retrieval regression tests are currently non-functional.
3. **[HIGH]** Fix `QueryCache.set` serialization for `RankedChunk` — Redis query cache is silently broken in production.
4. **[HIGH]** Fix `_vector_search` score: use `max(0.0, 1.0 - dist/2.0)` or configure ChromaDB collection for cosine distance.
5. **[MEDIUM]** Add `min_words` enforcement to `_fixed_chunk` to prevent stub trailing chunks.
6. **[MEDIUM]** Upgrade chunk ID hashing from MD5 to SHA-256 with 100-char text prefix.
7. **[MEDIUM]** Cache synonym map in `_expand_query` to eliminate O(N) per-query graph traversal.
8. **[LOW]** Move `logger` definition above `_get_collection()` in `retriever.py`.
9. **[Enhancement]** Add `entities` metadata field during ingest to enable the metadata filter path in `_entity_search`.
10. **[Enhancement]** Add fixture chunks with `entities` field and test the metadata filter code path.

---

*Report generated by: Test Agent 1 — RAG Pipeline Specialist*
*Analysis method: Static code analysis + manual metric computation from fixture data*
*Shell execution was unavailable; pytest output is derived from static analysis. Metric values are computed against `tests/eval/fixtures/chunks.json` and `tests/eval/golden/dataset.json`.*
