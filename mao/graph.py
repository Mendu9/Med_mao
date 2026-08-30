"""
mao/graph.py
------------
Assembles the LangGraph StateGraph connecting all MAO agents.

Graph topology:

    START
      |
      v
    chitchat_gate ------------------------------------.
      |  (not chitchat)                (chitchat)      |
      v                                                |
    decomposer  ->  classifier  ->  router_node        |
                                        |              |
                                        v              |
                                    risk_gate <--------'
                                        |
                        [conditional dispatch on state["intent"]]
                                        |
      .-------------------+-------------+-------------+-------------.
      v                   v             v             v             v
  summarizer_node   graphrag_node   tool_node   clinical_node   ...
  critic_node       clinical_node   chitchat_node
      |                   |             |             |             |
      '-------------------+------ verification -------+-------------'
                                        |
                                        v
                              domain_supervisor
                                        |
                                        v
                                     council
                                   /         \\
                        senior_supervisor    blocked
                                   \\         /
                                       END

Invariants this topology exists to enforce:

  1. `risk_gate` is the ONLY edge into an agent node. Every request therefore
     carries `state["risk_level"]` — computed from the safety policy — before
     any agent runs. Agents read it to decide what they may skip; nothing may
     reach an agent without it.

  2. `verification` is the ONLY edge from an agent node into the supervision
     chain. No agent can reach `domain_supervisor` without being verified
     first, whether the request was streamed or not (P0-1).

  3. Chitchat is detected exactly once, at `chitchat_gate` (P2-14). The gate
     short-circuits the decomposer/classifier/router pipeline but still passes
     through `risk_gate`, so invariant 1 holds on that path too.

  4. There is no SQL route (P0-3): LLM-authored SQL against the operational
     database was removed rather than sandboxed.

  5. There is no top-level multimodal node. Modality is a capability of
     `clinical_node`, which owns image, report and audio. The router forces every
     attachment to `clinical`, so a separate multimodal node could only ever be
     entered with nothing attached — it had no way to receive the data it existed
     to process.

After `senior_supervisor` or `blocked` the graph terminates. Multi-turn state is
managed by the API layer, which reinvokes the graph per request with updated
chat_history — that keeps the graph deterministic and debuggable.

Usage:
    from mao.graph import build_graph
    graph = build_graph()
    result = graph.invoke(make_initial_state("Who is Einstein?", "u1"))
    print(result["response"])
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from mao.agents.chitchat_agent import chitchat_node
from mao.agents.clinical_agent import clinical_node
from mao.agents.critic_agent import critic_node
from mao.agents.domain_classifier import classifier_node
from mao.agents.domain_supervisor import domain_supervisor_node
from mao.agents.graphrag_agent import graphrag_node
from mao.agents.llm_council import REVIEW_UNAVAILABLE, council_node
from mao.agents.query_decomposer import decomposer_node
from mao.agents.router import is_chitchat, route_to_agent, router_node
from mao.agents.senior_supervisor import senior_supervisor_node
from mao.agents.summarizer_agent import summarizer_node
from mao.agents.tool_agent import tool_node
from mao.core.state import INTENT_CHITCHAT, MAOState
from mao.memory.interface import recall_node
from mao.safety.policy import RiskLevel, get_policy, has_attachment

# Imported at module scope on purpose. Verification is a safety control: if it
# cannot load, the correct behaviour is to refuse to build the graph, not to
# serve unverified answers. A lazy import inside the node could not tell
# "not integrated yet" from "installed but its own dependencies are broken on
# this host" — and on a CPU-constrained deployment the second case is real.
from mao.safety.verification import verification_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node names — kept as constants to avoid typo bugs
# ---------------------------------------------------------------------------

NODE_GATE       = "chitchat_gate"
NODE_ROUTER     = "router_node"
NODE_RISK       = "risk_gate"
NODE_RECALL     = "recall"
NODE_VERIFY     = "verification"
NODE_SUMMARIZER = "summarizer_node"
NODE_GRAPHRAG   = "graphrag_node"
NODE_TOOL       = "tool_node"
NODE_CRITIC     = "critic_node"
NODE_CLINICAL   = "clinical_node"
NODE_CHITCHAT   = "chitchat_node"

_ALL_AGENT_NODES = [
    NODE_SUMMARIZER,
    NODE_GRAPHRAG,
    NODE_TOOL,
    NODE_CRITIC,
    NODE_CLINICAL,
    NODE_CHITCHAT,
]

# ---------------------------------------------------------------------------
# Pipeline nodes owned by the graph itself
# ---------------------------------------------------------------------------

def chitchat_gate_node(state: dict) -> dict:
    """Zero-cost entry gate — the single chitchat detection site (P2-14).

    Marking the intent here lets the conditional edge skip the decomposer,
    the domain classifier, and the router LLM call entirely.

    An attachment vetoes the shortcut. `is_chitchat` matches bare tokens like
    "ok", "no" and "yes", which are just as likely to be a clinician answering a
    follow-up question while attaching a scan. Short-circuiting those to the
    canned greeting discarded the attachment silently — no error, no log, and a
    reply that never mentions the scan the user just uploaded.
    """
    if has_attachment(state.get("metadata")):
        return state
    if is_chitchat(state.get("user_query", "")):
        return {**state, "intent": INTENT_CHITCHAT}
    return state


def _route_from_gate(state: dict) -> str:
    """Chitchat goes straight to the risk gate; everything else recalls first.

    Chitchat skips recall deliberately: a greeting needs no user history, and
    the shortcut exists precisely to avoid paying for work the answer cannot use.
    """
    return NODE_RISK if state.get("intent") == INTENT_CHITCHAT else NODE_RECALL


def risk_gate_node(state: dict) -> dict:
    """Write ``state["risk_level"]`` before any agent node can run.

    Classification is the safety policy's job, not the graph's — this node only
    supplies the two inputs (routed intent, presence of an attachment) and
    stores the verdict as a plain string so it survives JSON serialisation at
    the API boundary.
    """
    attachment = has_attachment(state.get("metadata"))
    risk = get_policy().risk_for(state.get("intent", ""), has_attachment=attachment)

    # A router that could not classify leaves us blind, and the cheapest route
    # (graphrag) is also the one that carries no clinical disclaimer. Treat an
    # unclassified request as HIGH so it keeps the controls a clinical request
    # would have had, rather than silently declassifying it.
    if state.get("router_failed"):
        risk = RiskLevel.HIGH
        logger.warning("Risk gate: router classification failed → escalating to HIGH")

    logger.debug(
        "Risk gate: intent=%s attachment=%s → risk=%s",
        state.get("intent", ""), attachment, risk.value,
    )
    return {**state, "risk_level": risk.value}


def blocked_response_node(state: dict) -> dict:
    """Withhold the answer, and say accurately why.

    Two different facts reach this node, and they used to produce one message.
    A council that reviewed the answer and rejected it is a statement about the
    clinical content. A council that could not run — a provider rate limit, a
    timeout, an outage — is a statement about the system. Telling a clinician
    their answer "was flagged for patient safety" when a rate limiter fired
    asserts something about their patient that nothing established, and points
    them at the wrong response: one case warrants clinical caution, the other
    warrants pressing retry.
    """
    verdict = state.get("council_verdict") or {}
    blocked_by = verdict.get("blocked_by", "council")

    if blocked_by == REVIEW_UNAVAILABLE:
        message = (
            "The safety review could not be completed, so this response is being "
            "withheld. This is a system availability problem, not a finding about "
            "the clinical content. Please try again in a moment."
        )
    else:
        message = (
            f"I cannot provide this response. It was flagged by the {blocked_by} review "
            "for patient safety. Please consult a licensed clinician directly."
        )
    # `output_blocked` is the one signal that says "this is a withdrawal, not an
    # answer". `apply_output_guardrails` documents that every exit sets it, but
    # this node withheld the answer without doing so — leaving the flag False on
    # a withheld response, which is how the refusal ended up cached for 300s and
    # served to the retry the message itself invites.
    return {
        **state,
        "response": message,
        "output_blocked": True,
        "output_blocked_by": blocked_by,
    }


def _route_after_council(state: dict) -> str:
    """Route on the council's verdict, defaulting closed.

    Only an explicit ``passed is True`` proceeds. A missing, malformed, or
    non-boolean verdict routes to `blocked`: `run_council` already fails closed
    on infrastructure errors, and an optimistic default here would quietly undo
    that by treating a half-written verdict as approval.
    """
    verdict = state.get("council_verdict") or {}
    passed = verdict.get("passed") if isinstance(verdict, dict) else None
    return "senior_supervisor" if passed is True else "blocked"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the MAO LangGraph StateGraph.

    Returns a compiled graph ready for .invoke() or .astream().
    Call once at application startup and reuse the instance.
    """
    builder = StateGraph(MAOState)

    # --- Agent nodes ---
    builder.add_node(NODE_SUMMARIZER, summarizer_node)
    builder.add_node(NODE_GRAPHRAG,   graphrag_node)
    builder.add_node(NODE_TOOL,       tool_node)
    builder.add_node(NODE_CRITIC,     critic_node)
    builder.add_node(NODE_CLINICAL,   clinical_node)
    builder.add_node(NODE_CHITCHAT,   chitchat_node)

    # --- Pre-agent pipeline ---
    builder.add_node(NODE_GATE,   chitchat_gate_node)
    builder.add_node("decomposer", decomposer_node)
    builder.add_node("classifier", classifier_node)
    builder.add_node(NODE_ROUTER, router_node)
    builder.add_node(NODE_RISK,   risk_gate_node)
    builder.add_node(NODE_RECALL, recall_node)

    # --- Post-agent supervision pipeline ---
    builder.add_node(NODE_VERIFY,         verification_node)
    builder.add_node("domain_supervisor", domain_supervisor_node)
    builder.add_node("council",           council_node)
    builder.add_node("senior_supervisor", senior_supervisor_node)
    builder.add_node("blocked",           blocked_response_node)

    # --- Entry: chitchat_gate → (risk_gate shortcut | full pipeline) ---
    builder.add_edge(START, NODE_GATE)
    builder.add_conditional_edges(
        NODE_GATE,
        _route_from_gate,
        {NODE_RISK: NODE_RISK, NODE_RECALL: NODE_RECALL},
    )
    # Memory is recalled once, here — before the router (which classifies using
    # it) and before any agent. No agent re-fetches it.
    builder.add_edge(NODE_RECALL, "decomposer")
    builder.add_edge("decomposer", "classifier")
    builder.add_edge("classifier", NODE_ROUTER)

    # --- The router's verdict is priced by the policy before dispatch ---
    builder.add_edge(NODE_ROUTER, NODE_RISK)
    builder.add_conditional_edges(
        NODE_RISK,
        route_to_agent,
        {
            NODE_SUMMARIZER: NODE_SUMMARIZER,
            NODE_GRAPHRAG:   NODE_GRAPHRAG,
            NODE_TOOL:       NODE_TOOL,
            NODE_CRITIC:     NODE_CRITIC,
            NODE_CLINICAL:   NODE_CLINICAL,
            NODE_CHITCHAT:   NODE_CHITCHAT,
        },
    )

    # --- Every agent is verified before it reaches supervision (P0-1) ---
    for node_name in _ALL_AGENT_NODES:
        builder.add_edge(node_name, NODE_VERIFY)
    builder.add_edge(NODE_VERIFY, "domain_supervisor")

    builder.add_edge("domain_supervisor", "council")
    builder.add_conditional_edges(
        "council",
        _route_after_council,
        {
            "senior_supervisor": "senior_supervisor",
            "blocked":           "blocked",
        },
    )
    # Memory is NOT written here. The graph cannot see the output guardrails —
    # they run in the API layer after `graph.invoke` — and the NLI-ratio and
    # judge-score blocks live only there. Persisting from inside the graph
    # therefore committed text the guardrails went on to withdraw, and memory is
    # replayed as prompt context on later turns. `mao/api/finalize.py` owns the
    # order: guardrails first, then persist what survived them.
    builder.add_edge("senior_supervisor", END)
    builder.add_edge("blocked", END)

    compiled = builder.compile()
    logger.info("MAO graph compiled with %d nodes", len(compiled.nodes))
    return compiled


# ---------------------------------------------------------------------------
# Module-level singleton — imported by api/main.py
# ---------------------------------------------------------------------------

_graph = None


def get_graph():
    """Return the compiled graph singleton, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Visual debugging helper
# ---------------------------------------------------------------------------

def print_graph_structure() -> None:
    """Print the Mermaid diagram of the graph for documentation/debugging."""
    graph = build_graph()
    try:
        print(graph.get_graph().draw_mermaid())
    except Exception:  # noqa: BLE001
        # draw_mermaid not available in all LangGraph versions
        print(f"Nodes: {sorted(graph.nodes)}")
        print(
            "Edges: chitchat_gate → [decomposer → classifier → router_node] → "
            "risk_gate → [conditional] → agent → verification → "
            "domain_supervisor → council → (senior_supervisor | blocked) → END"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_graph_structure()

    # Smoke test with a stub
    from mao.core.state import make_initial_state
    graph = get_graph()
    state = make_initial_state("What is machine learning?", "user-smoke-test")
    result = graph.invoke(state)
    print("\nResponse:", result.get("response", "")[:200])
    print("Agent used:", result.get("agent_used"))
    print("Intent:", result.get("intent"))
    print("Risk:", result.get("risk_level"))
