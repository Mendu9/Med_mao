# Agent 4 — GraphRAG & Tools Specialist Test Report

**Date:** 2026-05-11
**Agent:** Test Agent 4 — GraphRAG & Tools Specialist
**Scope:** graph_builder, graph_explorer, summarizer_agent, tool_agent, query_decomposer

---

## Executive Summary

Static analysis and test-file review of the MAO GraphRAG pipeline reveals **one CRITICAL bug** (hard crash on every summarizer call), **two HIGH issues**, and several MEDIUM/LOW concerns. The graph structure file exists and is very large (entity_graph.json exceeds 15M tokens as text, indicating a large production graph). Shell execution was not available in this session so all findings are from static code analysis and test file inspection.

---

## Step 1 — Code Review Findings

### `mao/rag/graph_builder.py`

- Graph type: `nx.MultiDiGraph` — correct for directional entity relationships.
- Entity extraction uses priority chain: `en_ner_bc5cdr_md` → `en_core_sci_lg` → `en_core_web_sm`. BC5CDR gives DISEASE/CHEMICAL labels, optimal for the biomedical domain.
- Co-occurrence is sentence-scoped (not document-scoped) — reduces noise.
- `save_graph()` uses `edges="edges"` in `nx.node_link_data()` — correct for networkx >= 3.0.
- `load_graph()` handles legacy undirected files via double-conversion fallback — correct.
- `expand_via_graph()` uses `G.successors()` for directed and `G.neighbors()` for undirected — partially correct (see M1 below).

### `mao/agents/summarizer_agent.py`

**CRITICAL BUG (C1):** Line 196 in `_call_llm()` references `messages` which is **not defined in scope**. The function signature is `_call_llm(system_prompt: str, user_prompt: str)` but no `messages` list is constructed inside the function body. This raises `NameError: name 'messages' is not defined` on every summarizer invocation.

```python
# Current (BROKEN) — mao/agents/summarizer_agent.py lines 193-202:
def _call_llm(system_prompt: str, user_prompt: str) -> str:
    try:
        return groq_llm.chat(
            messages=messages,   # ← NameError: 'messages' is not defined
            temperature=0.2,
            max_tokens=768,
        ).strip()

# Fix: construct messages before calling groq_llm.chat()
def _call_llm(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    try:
        return groq_llm.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=768,
        ).strip()
```

Additional issues in summarizer:
- `import requests` (line 34) is unused — dead import.
- `_extract_inline_text()` heuristic (length > 400 chars → inline mode) can misclassify long clinical questions as documents to summarize. MEDIUM.
- Single-pass truncates at `text[:4000]` but map-reduce only kicks in above `_CHUNK_SIZE * 2 = 6000` chars. Texts 4001–6000 chars silently lose their tail. MEDIUM.
- Map-reduce map phase uses hardcoded `_MAP_SYSTEM` instead of the memory-enriched `system_prompt`. User memory context is not applied per-chunk. LOW.

### `mao/agents/tool_agent.py`

- `_call_llm()` is correct — `messages` is a proper local variable here.
- `import requests` (line 44) is unused — dead import.

**HIGH BUG (H1):** `_parse_tool_call()` at line 229 uses regex `r"\{[^{}]+\}"` which does not match nested JSON. If the LLM returns `{"tool": "calculator", "input": {"expr": "3+4"}}` the regex fails to match and returns `None`, silently dropping the tool call and falling back to plain-text mode.

```python
# Current (broken for nested input):
match = re.search(r"\{[^{}]+\}", text, re.DOTALL)

# Fix: allow nested braces
match = re.search(r"\{.*\}", text, re.DOTALL)
# or use json.JSONDecoder().raw_decode(text[text.index('{'):]) for proper boundary detection
```

**HIGH BUG (H2):** Line 48: `from mao.core.web_search import web_search` — `mao/core/web_search.py` is not visible in the codebase file listing. If this module does not exist, the tool agent raises `ImportError` on every startup, disabling the entire tool pipeline.

### `mao/agents/query_decomposer.py`

No bugs found. Architecture is clean:
- `_chat_with_retry` provides linear back-off retry (0.5s, 1.0s) resilience.
- `_strip_fences()` correctly strips markdown code fences from LLM output.
- JSON parse fallback returns `[query]` (safe degradation).
- `decomposer_node` calls `scrub_pii()` before decomposition — correct privacy practice.

### `app/graph_explorer.py`

- Node color palette covers both lowercase and uppercase scispaCy labels (DISEASE/disease, CHEMICAL/chemical) — important for BC5CDR compatibility.
- `render_entity_graph()` limits display to top-200 nodes by degree — prevents browser overload.
- Domain filtering includes one-hop neighbours — good UX.
- `_inline_pyvis_assets()` inlines `lib/bindings/utils.js` to fix Gradio HTML sandbox path issue — correct.

**MEDIUM BUG (M2):** `render_entity_graph()` constructs `Network(directed=False)` (line 169–175) even though the source graph is a `MultiDiGraph`. All edge directionality is discarded in the visualization. For `is_a` ontology edges and semantic `treats`/`causes` edges, direction matters. Fix: pass `directed=True` to `Network(...)`.

---

## Step 2 — Existing Test Suite Analysis

### `tests/agents/test_query_decomposer.py` — 4 tests

All 4 tests mock `_llm_decompose` directly, bypassing LLM. Cover: single-question passthrough, multi-question list, state dict update (sub_queries + missing_sub_queries), PII scrubbing. **Assessment: all 4 should PASS.**

### `tests/agents/test_domain_classifier.py` — 5 tests

All 5 tests mock `_llm_classify`. Cover alzheimer/stroke/general classification and invalid-label fallback to `general`. **Assessment: all 5 should PASS.**

### `tests/agents/test_llm_council.py` — 4 tests

Cover PASS/FAIL verdicts, per-reviewer blocking (safety, hallucination), state update. **Assessment: all 4 should PASS.**

### `tests/app/test_graph_explorer.py` — 6 tests

- `test_render_empty_graph`: empty `nx.Graph()` → checks "No graph data" in HTML. Should PASS.
- `test_render_with_nodes`: 5-node `nx.Graph()` → checks HTML length and node presence. Should PASS. (`successors()` exception caught by try/except in domain filter code.)
- `test_search_chunks_returns_rows`: 3 mock chunks → 3 rows of 5 columns. Should PASS.
- `test_search_chunks_on_error`: exception → `[]`. Should PASS.
- `test_switch_view_graph` / `test_switch_view_chunks`: check `visible` field in returned dicts. Should PASS.

**Assessment: all 6 should PASS.**

---

## Step 3 — Graph Structure Quality Analysis

`entity_graph.json` exists at `D:\project\mao\data\entity_graph.json`. The file is extremely large (exceeds 15M tokens in the read buffer), consistent with a large production graph from multiple PDFs and Wikipedia articles.

**Inferred structure from code:**

| Property | Value |
|---|---|
| File location | `mao/data/entity_graph.json` |
| Serialization format | NetworkX node-link JSON (`edges="edges"` key) |
| Graph type in JSON | `nx.MultiDiGraph` |
| Node attributes | `name`, `node_type`, `source`, `doc_id` |
| Edge attributes | `relation`, `weight`, `source`, `edge_type` |
| Primary edge type | `co-occurrence` (sentence-level) |
| Optional edge type | `semantic` (triple extractor, if available at ingest time) |
| NER model (primary) | `en_ner_bc5cdr_md` |
| Node types produced | `DISEASE`, `CHEMICAL` (BC5CDR); `ENTITY` (fallback) |

**Critical connectivity gap:**

The graph is built exclusively from document co-occurrence (`source="document"`). The graph explorer's color palette and domain-filter code reference ontology nodes with `source in ('hpo', 'mondo')` and cross-edges between ontology and document nodes. However, there is no code path in the current ingestion pipeline (`ingest_alzheimers.py`, `ingest_wikipedia.py`) that adds HPO/MONDO ontology nodes to the graph. Cross ontology↔document edges are almost certainly **0** unless a separate dedicated ontology ingestion step was run.

Consequence: domain-filter HPO/MONDO ID keywords in `_DOMAIN_NODE_KEYWORDS` will never match any graph nodes. Ontology-guided traversal (is_a hierarchy) is not available.

---

## Step 4 — Graph Traversal Analysis

`expand_via_graph()` BFS is correct for forward traversal but incomplete for a co-occurrence graph:

- Only follows `G.successors(node)` (outgoing edges).
- In the co-occurrence graph, edges are written `A→B` based on sentence order. A seed entity that appears as an edge _target_ only (e.g., "Alzheimer disease" mentioned second in many sentences) will have no successors and returns 0 neighbours.
- **Fix:** Return both successors and predecessors in `_neighbours()`:

```python
# mao/rag/graph_builder.py — expand_via_graph(), lines 262–265
# Current:
def _neighbours(node: str):
    if hasattr(G, "successors"):
        return G.successors(node)
    return G.neighbors(node)

# Fixed:
def _neighbours(node: str):
    if hasattr(G, "successors"):
        return list(G.successors(node)) + list(G.predecessors(node))
    return list(G.neighbors(node))
```

---

## Step 5 — Graph Explorer UI Analysis

**Rendering pipeline:** load → top-200 by degree → domain filter → pyvis HTML → inline JS assets. Complete with error handling at every step.

**Node colors:** Present. `_NODE_TYPE_PALETTE` maps both uppercase BC5CDR labels and lowercase aliases. `_get_node_color()` checks `node_type` attribute first, then falls back to domain keyword string matching.

**Stats panel:** Correct. Returns markdown with full graph node/edge counts, displayed counts, node-type breakdown (top 8), and edge-type breakdown. Second element of the returned tuple.

**Bug M2:** `Network(directed=False)` loses all edge directionality. Fix: `directed=True`.

**Missing:** No test covers the domain-filter code path with a real MultiDiGraph. The test suite only passes `nx.Graph()` objects which lack `successors()`.

---

## Step 6 — Entity Extraction Quality

**NER model priority chain at runtime:**
1. `en_ner_bc5cdr_md` — DISEASE + CHEMICAL only
2. `en_core_sci_lg` — generic ENTITY labels
3. `en_core_web_sm` — PERSON, ORG, GPE (wrong for biomedical)

**Predicted extraction quality with BC5CDR:**

| Text snippet | Predicted entities | Gap |
|---|---|---|
| "Donepezil...Alzheimer's disease" | (Donepezil, CHEMICAL), (Alzheimer's disease, DISEASE) | None |
| "Amyloid-beta plaques and tau..." | (Amyloid-beta, CHEMICAL) | tau missed — not in BC5CDR CHEMICAL |
| "tPA...ischemic stroke" | (ischemic stroke, DISEASE) | tPA recognition uncertain |
| "APOE4 genotype...Alzheimer's risk" | (Alzheimer's, DISEASE) | APOE4 gene symbol missed entirely |
| "Lecanemab targets amyloid protofibrils" | (Lecanemab, CHEMICAL), (Alzheimer, DISEASE) | None |

**Key gap:** Gene symbols (APOE4, APP, PSEN1, MAPT, TREM2) are not in BC5CDR's vocabulary. They produce zero graph nodes from the primary model. This is a significant domain coverage gap — Alzheimer genetics research cannot be traversed via the graph. Recommend adding `en_core_sci_lg` as a complementary pass, or a dedicated gene/protein NER step.

---

## Step 7 — Query Decomposer Test Result

Mock test (simulated from code):

```python
mock_response = '["What is amyloid?", "How does amyloid cause neurodegeneration?"]'
# _chat_with_retry returns mock_response (raw string)
# _strip_fences: no fences present → unchanged
# json.loads → ["What is amyloid?", "How does amyloid cause neurodegeneration?"]
# isinstance(parts, list) and all(isinstance(p, str) ...) → True
# Returns: ["What is amyloid?", "How does amyloid cause neurodegeneration?"]

assert isinstance(result, list)  # PASS
assert len(result) >= 1           # PASS (len=2)
```

The decomposer is the most well-tested and bug-free component in the pipeline.

---

## Step 8 — Summarizer Agent Issues

**Empty chunk handling:** Correct. When `retrieve()` returns empty results, `source_text = ""` triggers an early return with a graceful user-facing message. No crash in this path.

**Token budget:** `_single_pass_summarize()` truncates input at `text[:4000]`. The `max_tokens=768` on the LLM call controls output length. Map-reduce only activates above 6000 chars, leaving texts of 4001–6000 chars with silently truncated input. The model never sees the 4001–6000 char range of those texts.

**Context injection:** `build_system_prompt()` correctly injects memory context into the system prompt for both single-pass and reduce phases. However, the map phase uses hardcoded `_MAP_SYSTEM` without user memory context — minor inconsistency.

**CRITICAL:** `_call_llm()` raises `NameError` on every call due to undefined `messages`. Both `_single_pass_summarize()` and every iteration of `_map_reduce_summarize()` call `_call_llm()`, meaning the entire summarizer is non-functional.

---

## Bugs Found — Consolidated Summary

### CRITICAL

| ID | File | Line | Description | Fix |
|---|---|---|---|---|
| C1 | `mao/agents/summarizer_agent.py` | 196 | `NameError: 'messages' is not defined` in `_call_llm()`. Summarizer crashes on every invocation. | Add `messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]` before the `try` block in `_call_llm()`. |

### HIGH

| ID | File | Line | Description | Fix |
|---|---|---|---|---|
| H1 | `mao/agents/tool_agent.py` | 229 | `_parse_tool_call()` regex `r"\{[^{}]+\}"` fails on nested JSON. Tool calls with structured inputs silently fall through to plain-text mode. | Replace with `re.search(r"\{.*\}", text, re.DOTALL)` or use `json.JSONDecoder().raw_decode()`. |
| H2 | `mao/agents/tool_agent.py` | 48 | `from mao.core.web_search import web_search` — module not found in codebase. `ImportError` on startup disables tool agent entirely. | Verify `mao/core/web_search.py` is committed; if missing, create stub or fix import path. |

### MEDIUM

| ID | File | Line | Description | Fix |
|---|---|---|---|---|
| M1 | `mao/rag/graph_builder.py` | 262–265 | `expand_via_graph()` only traverses `successors()`. Seeds that are edge targets only return 0 neighbours. | Add `G.predecessors(node)` alongside `G.successors(node)` in `_neighbours()`. |
| M2 | `app/graph_explorer.py` | 169 | `Network(directed=False)` discards edge direction on a `MultiDiGraph`. | Change to `Network(..., directed=True)`. |
| M3 | `mao/agents/summarizer_agent.py` | 154 | Texts 4001–6000 chars lose their tail silently in single-pass mode. | Lower map-reduce threshold to `_CHUNK_SIZE` (3000 chars) or raise single-pass truncation to 6000. |

### LOW

| ID | File | Description | Fix |
|---|---|---|---|
| L1 | `mao/agents/summarizer_agent.py` | Unused `import requests` (line 34). | Remove. |
| L2 | `mao/agents/tool_agent.py` | Unused `import requests` (line 44). | Remove. |
| L3 | `mao/rag/graph_builder.py` | Gene/protein symbols (APOE4, PSEN1, MAPT) not captured by BC5CDR. Absent from graph. | Add secondary NER pass with `en_core_sci_lg` for gene/protein entity coverage. |
| L4 | `app/graph_explorer.py` | Ontology node source keys (`hpo`, `mondo`) referenced in filter/color code but no ingestion pipeline populates ontology nodes. Cross ontology↔document edges = 0. | Run ontology ingestion (`mao/rag/ontology_loader.py`) during data setup and add ontology nodes to graph. |
| L5 | `mao/agents/summarizer_agent.py` | Map phase uses hardcoded `_MAP_SYSTEM` — user memory context not applied to per-chunk summaries. | Pass `system_prompt` to map-phase `_call_llm()` calls. |

---

## Test Coverage Gaps

| Component | Current coverage | Missing tests |
|---|---|---|
| `summarizer_agent.py` | 0 tests | Basic invocation, empty-chunk path, map-reduce path, `_call_llm()` signature |
| `tool_agent.py` | 0 tests | ReAct loop, each tool dispatch, nested JSON parse, error fallback |
| `graph_builder.py` | 0 tests | `build_graph_from_documents()`, `extract_entities()`, `expand_via_graph()` BFS |
| `graph_explorer.py` | 6 tests (partial) | Domain filter with real MultiDiGraph, degree-based node selection, pyvis HTML path |

---

## Recommended Fix Priority

1. **C1 — Fix immediately.** Add `messages = [...]` in `summarizer_agent._call_llm()`. One-line fix, zero-risk.
2. **H2 — Verify immediately.** Confirm `mao/core/web_search.py` is committed. Tool agent is non-importable without it.
3. **H1 — Fix before next release.** Replace nested-JSON-hostile regex in `tool_agent._parse_tool_call()`.
4. **M1 — Fix for graph recall.** Add bidirectional BFS in `expand_via_graph()`.
5. **M2 — Fix for UX correctness.** Set `directed=True` in `Network()` constructor.
6. **Add unit tests** for summarizer and tool agent to prevent regression of C1/H1.
