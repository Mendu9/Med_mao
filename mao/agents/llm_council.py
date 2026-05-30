from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from mao.core.config import CLINICAL_MODEL, COUNCIL_MAX_TOKENS, COUNCIL_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_AGENTS = {
    "accuracy": (
        "You are the Clinical Accuracy agent. Assess whether the response is medically accurate "
        "and consistent with the retrieved context. Output exactly: VERDICT: PASS or VERDICT: FAIL, "
        "followed by one sentence of reasoning."
    ),
    "hallucination": (
        "You are the Hallucination Guard. Check if any claim in the response is not grounded in the "
        "provided context. Output exactly: VERDICT: PASS or VERDICT: FAIL, followed by one sentence."
    ),
    "safety": (
        "You are the Safety and Ethics agent. Check if the response could cause patient harm, "
        "violates medical ethics, or provides dangerous advice. Output exactly: VERDICT: PASS or "
        "VERDICT: FAIL, followed by one sentence."
    ),
}

def _call_agent(role: str, system: str, response: str, context: str) -> str:
    from mao.core.llm import chat
    prompt = f"CONTEXT:\n{context}\n\nRESPONSE TO EVALUATE:\n{response}"
    return chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=CLINICAL_MODEL,
        max_tokens=COUNCIL_MAX_TOKENS,
        temperature=0.0,
    ).strip()

def run_council(response: str, context: str) -> dict:
    verdicts: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_call_agent, role, system, response, context): role
            for role, system in _AGENTS.items()
        }
        try:
            for future in as_completed(futures, timeout=COUNCIL_TIMEOUT_SECONDS):
                role = futures[future]
                try:
                    verdicts[role] = future.result()
                except Exception as e:
                    logger.error("Council agent %s failed: %s", role, e)
                    verdicts[role] = "VERDICT: FAIL. Agent error."
        except FuturesTimeoutError:
            for role in _AGENTS:
                if role not in verdicts:
                    logger.error("Council agent %s timed out", role)
                    verdicts[role] = "VERDICT: FAIL. Timeout."

    if "FAIL" in verdicts.get("safety", ""):
        return {**verdicts, "passed": False, "blocked_by": "safety"}
    if "FAIL" in verdicts.get("hallucination", ""):
        return {**verdicts, "passed": False, "blocked_by": "hallucination"}

    fail_count = sum(1 for v in verdicts.values() if "FAIL" in v)
    return {**verdicts, "passed": fail_count < 2, "blocked_by": None}

def council_node(state: dict) -> dict:
    answer = state.get("response", state.get("answer", ""))
    chunks = state.get("retrieved_docs", [])

    # No retrieved context — skip hallucination/accuracy checks, pass by default
    if not chunks:
        verdict = {"passed": True, "blocked_by": None, "skipped": "no_context"}
        return {**state, "council_verdict": verdict}

    context = "\n".join(str(c) for c in chunks[:4])
    verdict = run_council(response=answer, context=context)
    return {**state, "council_verdict": verdict}
