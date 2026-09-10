"""Shared harness for driving the real graph at its real boundary.

Wave 6 established that unit-level green hides boundary defects: every judge
test called `verification_node` directly, so nothing noticed that `MAOState`
did not declare `judge_scores` and LangGraph therefore discarded it at the node
boundary. The control was correct code with no reachable call path.

This harness exists so a test can assert on **what `graph.invoke` actually
returns**. It scripts the provider rather than the nodes: every node runs its
real code, its real registered prompt, and its real parser — only the network
hop is replaced. A test that stubs `verification_node` proves nothing about
whether the graph keeps what the node returns.

`ScriptedProvider` also records the `max_tokens` each *named prompt* was called
with, which is what makes budget adequacy assertable against a call site rather
than against `inspect.getsource` string matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from mao.providers.llm.base import ProviderResponse

# --------------------------------------------------------------------------
# Default replies, keyed by registered prompt name. A test overrides only the
# one it is about.
# --------------------------------------------------------------------------

_GROUNDED_ANSWER = (
    "Donepezil is an acetylcholinesterase inhibitor. It raises synaptic "
    "acetylcholine and is licensed for mild to moderate Alzheimer's disease. "
    "Benefit is symptomatic rather than disease-modifying."
)

DEFAULT_REPLIES: dict[str, str] = {
    "router.classify": "clinical",
    "domain.classify": "alzheimer",
    "decomposer.split": '["what does donepezil do?"]',
    "clinical.synthesis": _GROUNDED_ANSWER,
    "clinical.extraction": "{}",
    "clinical.vision": "A T1-weighted axial brain MRI.",
    "graphrag.synthesis": _GROUNDED_ANSWER,
    "summarizer.map": "summary",
    "summarizer.reduce": "summary",
    "summarizer.synthesis": "summary",
    "critic.review": "Looks reasonable.",
    "tool.react": "Final Answer: done",
    "multimodal.audio_transcript": "transcript",
    "judge.safety": '{"safety": 10, "groundedness": 10, "notes": "no concerns"}',
    "council.accuracy": "VERDICT: PASS. Accurate.",
    "council.hallucination": "VERDICT: PASS. Grounded.",
    "council.safety": "VERDICT: PASS. Safe.",
    "domain_supervisor.reconcile": '{"ungrounded_claims": []}',
    "senior_supervisor.completeness": '{"missing": []}',
}

# Prompts that are not an evaluator's own system template and so must not be
# dispatch keys. `router.user_turn` is a user turn. `verifier.protocol` is the
# ADV15-10 framing appended to every council/judge system message; it is longer
# than the member templates, and dispatch takes the longest match.
_NOT_DISPATCHABLE = frozenset({"router.user_turn", "verifier.protocol"})


@dataclass
class ProviderCall:
    """One completion, resolved back to the prompt that produced it."""

    prompt_name: str
    max_tokens: int
    system: str
    user: str


class ScriptedProvider:
    """A ChatProvider that answers by registered prompt name.

    Dispatch is by matching the registered prompt *template* inside the system
    message, so an agent that wraps its template (memory injection, extra
    framing) still resolves. The longest match wins, so one template being a
    prefix of another cannot mis-route.
    """

    name = "scripted"

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        from mao.prompts import registry

        self.replies = {**DEFAULT_REPLIES, **(replies or {})}
        self.calls: list[ProviderCall] = []
        self._templates = sorted(
            (
                (registry().get(n).template, n)
                for n in registry().names()
                if n not in _NOT_DISPATCHABLE
            ),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

    def resolve_prompt_name(self, system: str) -> str:
        for template, name in self._templates:
            if template and template in system:
                return name
        return "unknown"

    def complete(
        self,
        *,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        system = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        )
        user = next(
            (m.get("content", "") for m in messages if m.get("role") == "user"), ""
        )
        name = self.resolve_prompt_name(system)
        self.calls.append(
            ProviderCall(prompt_name=name, max_tokens=max_tokens, system=system, user=user)
        )
        text = self.replies.get(name, "")
        return ProviderResponse(text=text, input_tokens=10, output_tokens=10)

    # -- assertions the tests need ------------------------------------------

    def budget_for(self, prompt_name: str) -> int:
        """The `max_tokens` the *call site* used for this prompt."""
        for call in self.calls:
            if call.prompt_name == prompt_name:
                return call.max_tokens
        raise AssertionError(
            f"{prompt_name!r} was never called; saw {sorted({c.prompt_name for c in self.calls})}"
        )

    def called(self, prompt_name: str) -> bool:
        return any(c.prompt_name == prompt_name for c in self.calls)


class _NullMemory:
    def recall(self, query: str, user_id: str) -> str:
        return ""

    def remember(self, query: str, response: str, user_id: str) -> None:
        self.written = getattr(self, "written", [])
        self.written.append((query, response, user_id))


@dataclass
class GraphHarness:
    """Drive the real compiled graph with a scripted provider."""

    provider: ScriptedProvider
    memory: _NullMemory
    nli_flags: list[dict] = field(default_factory=list)
    retrieved: list[Any] = field(default_factory=list)

    def invoke(self, query: str = "what does donepezil do?", **overrides) -> dict:
        from mao.core.state import make_initial_state
        from mao.graph import build_graph

        state = make_initial_state(query, overrides.pop("user_id", "u-harness"))
        state.update(overrides)
        return build_graph().invoke(state)

    def finalize(self, result: dict, session_id: str = "sess-harness") -> dict:
        """Run the post-graph half of a request: guardrails, then memory."""
        import asyncio

        from mao.api.finalize import finalize_response

        return asyncio.run(finalize_response(result, session_id))


@pytest.fixture
def graph_harness(monkeypatch: pytest.MonkeyPatch):
    """The whole request path, offline, with every node running its real code.

    Only three things are replaced: the provider hop, retrieval, and the NLI
    cross-encoder. Everything the safety chain does with those answers is the
    production implementation.
    """
    from mao.agents import clinical_agent, graphrag_agent
    from mao.memory.interface import reset_memory_store, set_memory_store
    from mao.providers import gateway
    from mao.safety import verification

    provider = ScriptedProvider()
    gateway.set_provider(provider)

    memory = _NullMemory()
    set_memory_store(memory)

    harness = GraphHarness(provider=provider, memory=memory)

    monkeypatch.setattr(clinical_agent, "retrieve", lambda *a, **k: harness.retrieved)
    monkeypatch.setattr(clinical_agent, "_web_search_clinical", lambda q: "")
    monkeypatch.setattr(graphrag_agent, "retrieve", lambda *a, **k: harness.retrieved, raising=False)
    monkeypatch.setattr(
        verification, "check_all_claims", lambda claims, premise: harness.nli_flags
    )
    monkeypatch.setattr(
        "mao.guardrails.db_helper.log_guardrail_event",
        lambda *a, **k: None,
        raising=False,
    )

    yield harness

    gateway.reset_provider()
    reset_memory_store()


@pytest.fixture
def scripted_provider() -> Callable[..., ScriptedProvider]:
    """Bind a scripted provider without building the graph."""
    from mao.providers import gateway

    created: list[ScriptedProvider] = []

    def _make(replies: dict[str, str] | None = None) -> ScriptedProvider:
        provider = ScriptedProvider(replies)
        gateway.set_provider(provider)
        created.append(provider)
        return provider

    yield _make

    gateway.reset_provider()
