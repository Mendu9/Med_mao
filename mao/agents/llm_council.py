from __future__ import annotations

import asyncio
import logging
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


async def _async_call_agent(role: str, system: str, response: str, context: str) -> tuple[str, str]:
    """Call one council agent via async Groq. Returns (role, verdict_string)."""
    from mao.core.llm import achat
    prompt = f"CONTEXT:\n{context}\n\nRESPONSE TO EVALUATE:\n{response}"
    try:
        result = await asyncio.wait_for(
            achat(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                model=CLINICAL_MODEL,
                max_tokens=COUNCIL_MAX_TOKENS,
                temperature=0.0,
            ),
            timeout=COUNCIL_TIMEOUT_SECONDS,
        )
        return role, result.strip()
    except asyncio.TimeoutError:
        logger.error("Council agent %s timed out after %.1fs", role, COUNCIL_TIMEOUT_SECONDS)
        return role, "VERDICT: FAIL. Timeout."
    except Exception as exc:
        logger.error("Council agent %s failed: %s", role, exc)
        return role, "VERDICT: FAIL. Agent error."


async def run_council_async(response: str, context: str) -> dict:
    """Run all three council agents in parallel with asyncio.gather."""
    tasks = [
        _async_call_agent(role, system, response, context)
        for role, system in _AGENTS.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verdicts: dict[str, str] = {}
    for item in results:
        if isinstance(item, Exception):
            logger.error("Council gather exception: %s", item)
            continue
        role, verdict = item
        verdicts[role] = verdict

    # Ensure all roles have a verdict (fill missing with FAIL on gather exception)
    for role in _AGENTS:
        if role not in verdicts:
            verdicts[role] = "VERDICT: FAIL. Missing verdict."

    if "FAIL" in verdicts.get("safety", ""):
        return {**verdicts, "passed": False, "blocked_by": "safety"}
    if "FAIL" in verdicts.get("hallucination", ""):
        return {**verdicts, "passed": False, "blocked_by": "hallucination"}

    fail_count = sum(1 for v in verdicts.values() if "FAIL" in v)
    # Clinical pipeline: require unanimous PASS — any single FAIL blocks
    return {**verdicts, "passed": fail_count == 0, "blocked_by": None}


def run_council(response: str, context: str) -> dict:
    """Synchronous wrapper — runs the async council from a sync (executor thread) context.

    council_node is always called from a ThreadPoolExecutor thread (graph.invoke runs in executor).
    asyncio.run() creates a fresh event loop for that thread, drives the async gather, and closes it.
    This is the correct pattern for calling async code from a sync thread.
    """
    try:
        return asyncio.run(run_council_async(response, context))
    except Exception as exc:
        logger.error("Council run failed: %s", exc)
        # Fail-safe: a broken council infrastructure should NOT pass responses through
        return {"passed": False, "blocked_by": "council_error", "error": str(exc)}


def council_node(state: dict) -> dict:
    answer = state.get("response", state.get("answer", ""))
    chunks = state.get("retrieved_docs", [])

    # Skip supervision when streaming path deferred the LLM call — response is empty
    if state.get("_want_stream") and not answer:
        return {**state, "council_verdict": {"passed": True, "blocked_by": None, "skipped": "streaming_deferred"}}

    # No retrieved context — skip hallucination/accuracy checks, pass by default
    if not chunks:
        verdict = {"passed": True, "blocked_by": None, "skipped": "no_context"}
        return {**state, "council_verdict": verdict}

    context = "\n".join(str(c) for c in chunks[:4])
    verdict = run_council(response=answer, context=context)
    return {**state, "council_verdict": verdict}
