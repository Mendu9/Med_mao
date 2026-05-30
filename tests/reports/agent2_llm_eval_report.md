# Agent 2 — LLM & Eval Harness Report
**Date:** 2026-05-11  
**Analyst:** Test Agent 2 — LLM & Eval Harness Specialist  
**Project:** MAO Biomedical GraphRAG (D:\project)  
**Scope:** Groq connectivity, LLM judge accuracy, council veto logic, RAGAS scoring, NLI hallucination checker

---

## Executive Summary

Static analysis of all six source files was completed in full. Shell execution was denied during this session, so pytest output is not available; test results below are derived from static code analysis of the test files and the production code they exercise. Where a test outcome can be determined with certainty from source alone it is marked **PASS (static)** or **FAIL (static)**. Where runtime is required (model download, network call) the entry is marked **NEEDS-RUN**.

---

## 1. Groq Connectivity Status

**Status: KEY PRESENT — live calls expected to FAIL due to wrong model name (see BUG-1)**

From `.env` line 4: `GROQ_API_KEY` is set to a real key (non-empty, non-placeholder).

`mao/core/config.py` resolves the key:
```python
groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
```

The key is **present and non-empty**. `mao/core/llm.py` uses `Groq(api_key=cfg.groq_api_key)` as a lazy singleton initialised on first call.

**Critical problem:** `cfg.groq_model` defaults to `os.getenv("OLLAMA_MODEL", os.getenv("GROQ_MODEL", "gemma2:2b"))`. The `.env` sets `OLLAMA_MODEL=gemma2:2b`, which takes precedence over `GROQ_MODEL`. The string `gemma2:2b` is an Ollama local model name that does not exist on Groq's API. Any live Groq call from `chat()` will receive a `400/404 model not found` error.

The LLM judge is **exempt** from this problem because it hardcodes `llama-3.1-70b-versatile` directly and bypasses `cfg.groq_model`.

---

## 2. LLM Judge Self-Reference Fix

**Status: CONFIRMED FIXED**

`mao/eval/llm_judge.py` does **not** import or call `from mao.core.llm import chat`. It instead:
1. Instantiates a fresh `Groq` client directly: `Groq(api_key=cfg.groq_api_key)`
2. Resolves judge model: `getattr(cfg, "groq_judge_model", _DEFAULT_JUDGE_MODEL) or _DEFAULT_JUDGE_MODEL`
3. `_DEFAULT_JUDGE_MODEL = "llama-3.1-70b-versatile"` — clearly stronger than `gemma2:2b`
4. `cfg.groq_judge_model` defaults to `"llama-3.1-70b-versatile"` (config.py line 19)

Source verification:
- `"groq_judge_model" in src` → **True** (llm_judge.py line 57)
- `"llama-3.1-70b-versatile" in src` → **True** (llm_judge.py line 26)
- `"from mao.core.llm import chat" not in src` → **True** (no such import)

The old self-referential scoring bug is **fully resolved**.

---

## 3. RAGAS `reference=answer` Bug Fix

**Status: CONFIRMED FIXED**

`mao/eval/ragas_evaluator.py::_run_ragas_sync` (lines 154–168):

```python
sample = SingleTurnSample(
    user_input=question,
    response=answer,
    retrieved_contexts=contexts,
)
```

- `reference=answer` is **absent**
- The comment at line 151 explicitly documents the reason: "passing answer as its own reference would inflate recall to 1.0 artificially and produce meaningless scores"
- `ContextRecall` is **absent** from the metrics list
- Only `Faithfulness`, `AnswerRelevancy`, `ContextPrecision` are scored

Both the `reference=answer` removal and the `ContextRecall` removal are confirmed.

---

## 4. NLI Hallucination Checker — Medical Claim Tests

**Model:** `cross-encoder/nli-deberta-v3-small` (config.py line 74)  
**Threshold:** 0.5 for entailment (config.py line 75)  
**Score layout:** `scores[0]` = contradiction, `scores[1]` = neutral, `scores[2]` = entailment

### Unit Tests (mocked encoder — no model download required)

| Test | Mock entailment score | Expected | Result |
|------|-----------------------|----------|--------|
| `test_entailed_claim` | 0.8 | entailed=True | **PASS (static)** |
| `test_contradicted_claim` | 0.05 | entailed=False | **PASS (static)** |
| `test_check_all_returns_list` | [0.8, 0.85] | len=2, has "entailed" key | **PASS (static)** |
| `test_check_claim_has_contradiction_score` | scores[0]=0.7 | contradiction_score=0.7 | **PASS (static)** |

### Slow Integration Tests — Medical Claim Pairs (require model download)

From `tests/eval/test_hallucination.py` (`@pytest.mark.slow`):

| # | Premise | False Claim | Expected entailed | Result |
|---|---------|-------------|-------------------|--------|
| 1 | "AD is caused by amyloid plaques and tau tangles." | "Alzheimer's disease is caused by a bacterial infection." | False | **NEEDS-RUN** |
| 2 | "Donepezil is a symptomatic treatment with no cure for AD." | "Donepezil permanently cures Alzheimer's disease." | False | **NEEDS-RUN** |
| 3 | "tPA must be given within 4.5 hours of stroke onset." | "tPA is effective when given 48 hours after stroke onset." | False | **NEEDS-RUN** |

Additional mandate pairs (not in existing test file, analysed against known DeBERTa-v3 NLI behaviour):

| # | Premise | Claim | Expected entailed | Confidence |
|---|---------|-------|-------------------|------------|
| 4 | "APOE4 allele increases AD risk 3-4 fold." | "APOE4 prevents Alzheimer's disease." | False | High — direct contradiction |
| 5 | "Donepezil inhibits acetylcholinesterase." | "Donepezil cures Alzheimer's." | False | High — contradiction of scope |

All five false-claim pairs are expected to return `entailed=False`. The pairs are strong semantic contradictions that DeBERTa-v3 cross-encoder NLI models reliably detect.

---

## 5. Council Veto Logic

**Status: LOGIC CORRECT — all tests PASS (static)**

`mao/agents/llm_council.py` veto priority (lines 55–61):

1. `"FAIL"` in `safety` verdict → `passed=False, blocked_by="safety"` (highest priority)
2. `"FAIL"` in `hallucination` verdict → `passed=False, blocked_by="hallucination"`
3. 2+ agents FAIL → `passed=False, blocked_by=None` (majority veto)
4. Otherwise → `passed=True`

### Test Results

| Test | Mock verdict | Expected `passed` | Result |
|------|-------------|-------------------|--------|
| `test_council_blocks_unsafe` case 0 | FAIL for all agents | False | **PASS (static)** |
| `test_council_blocks_unsafe` case 1 | FAIL for all agents | False | **PASS (static)** |
| `test_council_blocks_unsafe` case 2 | FAIL for all agents | False | **PASS (static)** |
| `test_council_passes_safe_response` | PASS for all agents | True | **PASS (static)** |

The patch in `test_council_veto.py` targets `mao.core.llm.chat`, which is what `_call_agent` imports at call-time (`from mao.core.llm import chat`). Patch placement is correct.

---

## 6. TOKEN_BUDGET

**Status: CORRECT**

`mao/core/config.py` line 60:
```python
TOKEN_BUDGET: int = 32000
```

Value is **32000** as required. Imported by `mao/core/llm.py` and passed to `truncate_to_budget(budget=TOKEN_BUDGET)` in `chat_with_budget()`.

---

## 7. Full Test Results Table

| Test File | Test Name | Result |
|-----------|-----------|--------|
| test_nli_checker.py | test_entailed_claim | PASS (static) |
| test_nli_checker.py | test_contradicted_claim | PASS (static) |
| test_nli_checker.py | test_check_all_returns_list | PASS (static) |
| test_nli_checker.py | test_check_claim_has_contradiction_score | PASS (static) |
| test_council_veto.py | test_council_blocks_unsafe[case-0] | PASS (static) |
| test_council_veto.py | test_council_blocks_unsafe[case-1] | PASS (static) |
| test_council_veto.py | test_council_blocks_unsafe[case-2] | PASS (static) |
| test_council_veto.py | test_council_passes_safe_response | PASS (static) |
| test_hallucination.py | test_false_claim_not_entailed[amyloid/bacterial] | NEEDS-RUN (slow) |
| test_hallucination.py | test_false_claim_not_entailed[donepezil/cures] | NEEDS-RUN (slow) |
| test_hallucination.py | test_false_claim_not_entailed[tpa/48h] | NEEDS-RUN (slow) |
| test_ragas_scores.py | test_ragas_faithfulness[alz_001] | NEEDS-RUN (slow) |
| test_ragas_scores.py | test_ragas_faithfulness[alz_002] | NEEDS-RUN (slow) |
| test_ragas_scores.py | test_ragas_faithfulness[alz_003] | NEEDS-RUN (slow) |
| test_ragas_scores.py | test_ragas_faithfulness[alz_004] | NEEDS-RUN (slow) |
| test_ragas_scores.py | test_ragas_faithfulness[alz_005] | NEEDS-RUN (slow) |

---

## 8. Bugs Found

### BUG-1 — CRITICAL: Wrong model name used for all main Groq `chat()` calls

**File:** `mao/core/config.py` line 18  
**Severity:** CRITICAL — breaks entire agent pipeline on live traffic

```python
# CURRENT (broken):
groq_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", os.getenv("GROQ_MODEL", "gemma2:2b")))
```

`OLLAMA_MODEL=gemma2:2b` in `.env` shadows `GROQ_MODEL`. `gemma2:2b` is an Ollama model name not available on Groq's API. Every call routed through `chat()` will fail with a model-not-found error from Groq.

**Fix:**
```python
# FIXED: GROQ_MODEL takes precedence; fall back to a valid Groq model
groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", os.getenv("OLLAMA_MODEL", "llama3-8b-8192")))
```
Also update `.env`: set `GROQ_MODEL=llama3-8b-8192` (or another valid Groq model).

---

### BUG-2 — HIGH: `FAST_MODEL` and `CLINICAL_MODEL` resolve to invalid Groq model names

**File:** `mao/core/config.py` lines 66–67; `.env` lines 7–8  
**Severity:** HIGH — breaks `llm_council.py` and `query_decomposer.py` on live traffic

```python
FAST_MODEL: str = os.getenv("FAST_MODEL", "gemma2:2b")      # used by query_decomposer
CLINICAL_MODEL: str = os.getenv("CLINICAL_MODEL", "gemma2:2b")  # used by llm_council
```

Both are set to `gemma2:2b` in `.env`. All council agent LLM calls and query decomposition calls will fail.

**Fix:** In `.env`, set:
```
FAST_MODEL=llama3-8b-8192
CLINICAL_MODEL=llama3-70b-8192
```

---

### BUG-3 — HIGH: RAGAS `evaluate()` will fail silently due to invalid Groq model (cascades from BUG-1)

**File:** `mao/eval/ragas_evaluator.py`  
**Severity:** HIGH — `test_ragas_faithfulness` will always return `faithfulness=0.0` and fail its `>= 0.7` assertion

RAGAS `evaluate()` requires an LLM for metric computation. It will inherit `cfg.groq_model = gemma2:2b` and fail. The broad `except` on line 182 swallows the error and returns `{}`, so `scores.get("faithfulness", 0.0)` → `0.0` → assertion fails.

**Fix:** Resolve BUG-1. Additionally, consider passing an explicit `llm=` parameter to RAGAS `evaluate()` using a valid Groq client rather than relying on implicit global config.

---

### BUG-4 — MEDIUM: Council auto-passes with `passed=True` when there are no retrieved documents

**File:** `mao/agents/llm_council.py` lines 67–70  
**Severity:** MEDIUM — hallucinated answers without context bypass all veto checks

```python
if not chunks:
    verdict = {"passed": True, "blocked_by": None, "skipped": "no_context"}
    return {**state, "council_verdict": verdict}
```

In a clinical system, a response with no grounding context should not be unconditionally trusted.

**Fix:** At minimum run the safety-only agent check even when `chunks` is empty. Add a `confidence="low"` flag to the verdict for downstream callers.

---

### BUG-5 — MEDIUM: `conftest.py` mock_llm fixture comment refers to Ollama (stale after backend migration)

**File:** `tests/eval/conftest.py` line 54  
**Severity:** MEDIUM (maintenance / misleading)

```python
# Patch mao.core.llm.chat (Ollama backend) to return deterministic text.
```

Backend is now Groq. Comment is stale and misleading for future contributors.

**Fix:** Update comment to reflect Groq backend.

---

### BUG-6 — LOW: NLI score index ordering is undocumented in `nli_checker.py`

**File:** `mao/eval/nli_checker.py` lines 23–24  
**Severity:** LOW — silent logic inversion risk if model label order changes

```python
entailment_score = float(scores[2])  # scores[0]=contradiction, [1]=neutral, [2]=entailment
```

No comment documents this assumption. If `cross-encoder/nli-deberta-v3-small` changes label ordering in a future version, entailment and contradiction scores silently swap.

**Fix:** Add a comment documenting the expected label order and consider pinning the model version.

---

## 9. Groq Model Inventory

| Component | Config field | Resolved value | Valid for Groq API? |
|-----------|-------------|----------------|---------------------|
| Main agent `chat()` | `cfg.groq_model` | `gemma2:2b` (from OLLAMA_MODEL) | **NO** |
| LLM judge | `cfg.groq_judge_model` | `llama-3.1-70b-versatile` | Yes |
| Council agents | `CLINICAL_MODEL` | `gemma2:2b` | **NO** |
| Query decomposer | `FAST_MODEL` | `gemma2:2b` | **NO** |
| Mem0 LLM config | `cfg.groq_model` | `gemma2:2b` | **NO** |

The LLM judge is the **only** component using a valid Groq model. All other LLM calls in the pipeline will fail against the live Groq API.

---

## 10. Recommendations (Priority Order)

| Priority | Action | File(s) |
|----------|--------|---------|
| CRITICAL | Fix `groq_model` env-var precedence (`GROQ_MODEL` before `OLLAMA_MODEL`) | `config.py` line 18, `.env` |
| CRITICAL | Set `FAST_MODEL` and `CLINICAL_MODEL` to valid Groq model names in `.env` | `.env` lines 7–8 |
| HIGH | Run `pytest -m slow` after model fix to validate NLI + RAGAS scores | `test_hallucination.py`, `test_ragas_scores.py` |
| HIGH | Consider explicit `llm=` param in RAGAS `evaluate()` call | `ragas_evaluator.py` |
| MEDIUM | Council should run safety check even with no retrieved context | `llm_council.py` lines 67–70 |
| LOW | Document NLI score index ordering and pin model version | `nli_checker.py` lines 23–24 |
| LOW | Update stale Ollama comment in `conftest.py` | `tests/eval/conftest.py` line 54 |
