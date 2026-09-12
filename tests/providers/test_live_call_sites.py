"""Every call site, at its REAL prompt and REAL budget, parsed by its REAL parser.

This is the test the architecture review asked for, and the one that would have
caught both Wave 6 P0s.

What it replaces, and why those did not work:

  - `TestLiveProviderAvailability` probed every role with
    `"Reply with the single word: ok"` at `max_tokens=200`. A trivial prompt at
    a budget no safety call site uses. It passed at the exact SHA where 5 of 5
    ordinary clinical questions were being refused.

  - Budget adequacy was asserted by `inspect.getsource` string matching:

        assert "max_tokens=5" not in source
        assert "max_tokens=10" not in source

    which passes with `max_tokens=6`, never inspects the safety budgets at all,
    and cannot detect a budget that is too small for the model now bound to the
    role. It is a test of the text of the code, not of what the code does.

The three properties that make this different:

  1. The budget comes from the call site's own constant, imported. Changing the
     constant changes the test; there is no second copy to drift.
  2. The prompt is the registered one the call site really sends.
  3. The assertion is run through the call site's own PARSER. "Non-empty text"
     is not enough — a truncated reasoning model can return text that no parser
     can read, which every fail-closed control then treats as a refusal.

Marked `slow`, like the other network tests, so the default run stays hermetic.
Run with: pytest -m slow tests/providers/test_live_call_sites.py
"""
from __future__ import annotations

import pytest

from mao.prompts import get_prompt
from mao.providers import gateway
from mao.trust.classes import TrustClass
from mao.trust.egress.policy import EgressPurpose
from mao.providers.registry import ModelRole

pytestmark = pytest.mark.slow


def _skip_if_rate_limited(exc: Exception) -> None:
    """A provider quota is an environment condition, not a defect.

    Every other skip in this suite is environment-gated; a daily-token ceiling
    is the same kind of thing. It is re-raised as a skip rather than swallowed
    so the run still shows that the probe did not actually execute.
    """
    text = str(exc).lower()
    if "rate_limit" in text or "rate limit" in text or "429" in text:
        pytest.skip(f"provider quota exhausted, probe not executed: {str(exc)[:160]}")


def _probe(fn):
    """Run a live probe, converting a provider quota error into a skip."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        _skip_if_rate_limited(exc)
        raise


# A correct, safe, well-hedged clinical answer with a matching premise.
#
# The assertions below are about whether each reviewer ANSWERS IN CONTRACT at
# its real budget, never about which verdict it reaches. A strict reviewer that
# returns "VERDICT: FAIL" has reviewed the answer; a reviewer that returns an
# empty string has not, and only the second is a call-site defect.
PREMISE = (
    "Donepezil inhibits acetylcholinesterase, increasing synaptic acetylcholine. "
    "Licensed for mild-to-moderate Alzheimer's disease. Adverse effects include "
    "nausea, diarrhoea and bradycardia. Start 5 mg once daily, titrate to 10 mg "
    "after four weeks. Review periodically for continued benefit."
)
ANSWER = (
    "Donepezil is an acetylcholinesterase inhibitor licensed for mild to moderate "
    "Alzheimer's disease. It increases synaptic acetylcholine. Common adverse "
    "effects include nausea, diarrhoea and bradycardia. The usual regimen is 5 mg "
    "daily for four weeks, then 10 mg daily if tolerated. Treatment should be "
    "reviewed periodically and any change confirmed by the treating clinician."
)


def _complete(role: ModelRole, prompt_name: str, user: str, budget: int) -> str:
    return gateway.complete(
        role=role,
        messages=[
            {"role": "system", "content": get_prompt(prompt_name).template},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=budget,
        purpose=EgressPurpose.GENERAL_SYNTHESIS,
        trust_class=TrustClass.SAFE_DERIVED_TEXT,
    ).text


def _verifier_messages(prompt_name: str) -> list[dict]:
    """The real structured messages a council member and the judge now send.

    `council.user_turn` is gone. It concatenated the evidence and the answer
    into one user turn, so a retrieved chunk could forge the frame and redirect
    the review (ADV15-10). Probing the old single turn would probe a shape
    production no longer sends — and this file exists precisely so a probe
    cannot drift from its call site.
    """
    from mao.prompts.verification import build_verifier_messages

    return build_verifier_messages(
        instructions=get_prompt(prompt_name).template,
        evidence=PREMISE,
        response=ANSWER,
    )


def _complete_verifier(role: ModelRole, prompt_name: str, budget: int) -> str:
    return gateway.complete(
        role=role,
        messages=_verifier_messages(prompt_name),
        temperature=0.0,
        max_tokens=budget,
        purpose=EgressPurpose.GENERAL_SYNTHESIS,
        trust_class=TrustClass.SAFE_DERIVED_TEXT,
    ).text


# ---------------------------------------------------------------------------
# The safety chain — where both Wave 6 P0s lived
# ---------------------------------------------------------------------------

class TestTheSafetyJudgeCallSite:
    """The property under test is that the judge ANSWERS IN CONTRACT at its real
    budget — not that it reaches any particular verdict.

    Asserting `safety >= 5` on a chosen answer would pin the model's clinical
    judgement, which is neither stable nor this test's business, and would make
    a genuinely strict judge look like a broken budget. What blocker 3 actually
    broke is narrower and fully deterministic: the reply never arrived, so no
    parser could read it, and a safe answer was adjudicated as a failure to
    review. That is what is asserted here.
    """

    def test_the_judge_returns_a_score_its_own_parser_can_read(self) -> None:
        from mao.safety.verification import (
            _JUDGE_MAX_TOKENS,
            _UNREADABLE_SAFETY,
            _parse_judge,
        )

        raw = _probe(
            lambda: _complete_verifier(
                ModelRole.SAFETY_JUDGE, "judge.safety", _JUDGE_MAX_TOKENS
            )
        )
        scores = _parse_judge(raw)

        assert "unparseable" not in scores["notes"], (
            f"at max_tokens={_JUDGE_MAX_TOKENS} the judge produced nothing its "
            f"own parser could read, so every answer fails closed: {raw[:200]!r}"
        )
        assert scores["safety"] != _UNREADABLE_SAFETY or "0" in raw, (
            "the safety score defaulted to the fail-closed value without the "
            "judge actually returning 0"
        )

    def test_the_judge_is_not_truncated_at_its_real_budget(self) -> None:
        from mao.safety.verification import _JUDGE_MAX_TOKENS

        completion = _probe(
            lambda: gateway.complete(
                role=ModelRole.SAFETY_JUDGE,
                messages=_verifier_messages("judge.safety"),
                temperature=0.0,
                max_tokens=_JUDGE_MAX_TOKENS,
                purpose=EgressPurpose.GENERAL_SYNTHESIS,
                trust_class=TrustClass.SAFE_DERIVED_TEXT,
            )
        )
        assert not completion.truncated, (
            "the judge hit its ceiling — the declared reasoning overhead for "
            f"{completion.model_id} is too low"
        )
        assert completion.text.strip()


class TestTheCouncilCallSites:
    @pytest.mark.parametrize(
        "prompt_name", ["council.accuracy", "council.hallucination", "council.safety"]
    )
    def test_the_member_answers_in_its_declared_format(self, prompt_name: str) -> None:
        """PASS or FAIL — either is a review. Silence is not.

        `parse_member_verdict` requires an affirmative PASS and fails closed on
        anything else, so an empty completion blocks the answer. Asserting PASS
        here would pin the member's judgement; asserting that it emitted a
        verdict token at all is exactly the budget property, and it is what was
        false at the frozen SHA (5 of 18 members returned `out=200/200 raw=''`).
        """
        from mao.agents.llm_council import _VERDICT_RE
        from mao.core.config import COUNCIL_MAX_TOKENS

        raw = _probe(
            lambda: _complete_verifier(
                ModelRole.SAFETY_JUDGE, prompt_name, COUNCIL_MAX_TOKENS
            )
        )

        assert raw.strip(), (
            f"{prompt_name} returned an empty completion at "
            f"max_tokens={COUNCIL_MAX_TOKENS} — the whole budget went on the "
            "model's analysis channel, and a failure to review blocks the answer"
        )
        assert _VERDICT_RE.search(raw), (
            f"{prompt_name} answered outside its declared VERDICT format, which "
            f"the parser treats as a failure to review: {raw[:200]!r}"
        )


class TestTheSupervisionCallSites:
    """Wave 9 / B10 — the role comes from the module under test, not from here.

    These two probed `SAFETY_JUDGE` while Wave 7 had already moved both
    supervisors to `EXTRACTION_FAST`. So they exercised a model production never
    binds, at 2.4-3.1x the ceiling production actually gives them, and passed —
    in the file the gate rests on, testing the very defect class it was written
    to prevent.

    Importing `_ROLE` alongside `_MAX_TOKENS` means the probe cannot drift from
    the call site again: moving a supervisor to another role moves its probe.

    Wave 11 / NB4: this was the only class in the file not wrapped in `_probe`,
    so a daily-quota error here FAILED the run instead of skipping it, and a
    provider ceiling would have been reported as a call-site defect.
    """

    def test_the_domain_supervisor_returns_parseable_json(self) -> None:
        from mao.agents.domain_supervisor import (
            _MAX_TOKENS,
            _ROLE,
            _parse_ungrounded_claims,
        )

        raw = _probe(
            lambda: _complete(
                _ROLE,
                "domain_supervisor.reconcile",
                f"RAG CHUNKS:\n{PREMISE}\nDRAFT ANSWER:\n{ANSWER}",
                _MAX_TOKENS,
            )
        )
        assert raw.strip(), (
            f"empty completion at role={_ROLE.value} max_tokens={_MAX_TOKENS}"
        )
        assert "{" in raw, f"no JSON object in the reply: {raw[:200]!r}"
        _parse_ungrounded_claims(raw)  # must not raise

    def test_the_senior_supervisor_returns_parseable_json(self) -> None:
        from mao.agents.senior_supervisor import _MAX_TOKENS, _ROLE, _parse_missing

        raw = _probe(
            lambda: _complete(
                _ROLE,
                "senior_supervisor.completeness",
                f'SUB-QUESTIONS:\n["what does donepezil do?"]\n\nANSWER:\n{ANSWER}',
                _MAX_TOKENS,
            )
        )
        assert raw.strip(), (
            f"empty completion at role={_ROLE.value} max_tokens={_MAX_TOKENS}"
        )
        assert "}" in raw, (
            f"the JSON object was cut off before it closed, so the parser "
            f"silently returns []: {raw[:200]!r}"
        )
        _parse_missing(raw)



# ---------------------------------------------------------------------------
# The routing and synthesis call sites
# ---------------------------------------------------------------------------

class TestTheRoutingCallSites:
    def test_the_router_classifies_at_its_real_budget(self) -> None:
        from mao.agents.router import _CLASSIFY_MAX_TOKENS
        from mao.core.state import ALL_INTENTS

        raw = _probe(
            lambda: _complete(
                ModelRole.ROUTER_FAST,
                "router.classify",
                get_prompt("router.user_turn").render(
                    memory_context="",
                    history="(none)",
                    query="My patient has LMCI. What treatment options exist?",
                ),
                _CLASSIFY_MAX_TOKENS,
            )
        )
        labels = {t.strip(".,!?:;\"'") for t in raw.strip().lower().split()}
        assert labels & set(ALL_INTENTS), (
            f"the router produced no recognised intent at "
            f"max_tokens={_CLASSIFY_MAX_TOKENS}: {raw!r} — every request would "
            "be marked router_failed and escalated to HIGH"
        )

    def test_the_domain_classifier_classifies_at_its_real_budget(self) -> None:
        from mao.agents.domain_classifier import _DOMAIN_MAX_TOKENS, _VALID_DOMAINS

        raw = _probe(
            lambda: _complete(
                ModelRole.EXTRACTION_FAST,
                "domain.classify",
                "My patient has LMCI. What treatment options exist?",
                _DOMAIN_MAX_TOKENS,
            )
        )
        assert raw.strip().lower() in _VALID_DOMAINS, (
            f"unrecognised domain {raw!r} — every query silently becomes 'general'"
        )


def _synthesis_call_sites() -> list[tuple[ModelRole, str, int]]:
    """Budgets read from the modules that spend them.

    These were hardcoded as 1024 and 768. They matched the call sites at the
    time, with nothing keeping them matched — the same defect the supervisors
    above actually hit, one edit away.
    """
    from mao.agents.clinical_agent import _SYNTHESIS_MAX_TOKENS as _CLINICAL
    from mao.agents.graphrag_agent import _SYNTHESIS_MAX_TOKENS as _GRAPHRAG

    return [
        (ModelRole.CLINICAL_SYNTHESIS, "clinical.synthesis", _CLINICAL),
        (ModelRole.GENERAL_SYNTHESIS, "graphrag.synthesis", _GRAPHRAG),
    ]


class TestTheSynthesisCallSites:
    @pytest.mark.parametrize("role,prompt_name,budget_ref", _synthesis_call_sites())
    def test_synthesis_returns_a_substantive_answer(
        self, role: ModelRole, prompt_name: str, budget_ref: int
    ) -> None:
        raw = _complete(
            role,
            prompt_name,
            f"## Relevant Research\n{PREMISE}\n\nUser question: what does donepezil do?",
            budget_ref,
        )
        assert len(raw.strip()) > 200, (
            f"{prompt_name} produced {len(raw.strip())} chars at "
            f"max_tokens={budget_ref} — the answer budget is being spent on the "
            "model's analysis channel"
        )

    def test_the_clinical_synthesis_budget_answers_from_a_safe_projection(self) -> None:
        """The report path no longer summarises the document.

        `_REPORT_SUMMARY_MAX_TOKENS` budgeted a call that sent the de-identified
        report to an external model — a payload the approved M-1 policy does not
        admit. The probe is replaced rather than deleted: what needs a live
        budget check now is the typed synthesis, which is the only external call
        the migrated report path makes.
        """
        from mao.agents.clinical_agent import _SYNTHESIS_MAX_TOKENS
        from mao.trust.classes import PublicEvidence, SafeSynthesisContext

        context = SafeSynthesisContext(
            context_id="live-probe",
            clinical_question="Is it safe to continue donepezil?",
            age_group="older_adult_75_89",
            medications_and_doses=("donepezil 10 mg once daily",),
            clinically_relevant_findings=("bradycardia 48 bpm", "dizziness"),
        )
        completion = gateway.synthesise_clinical(
            system_prompt="You are a clinical decision-support assistant.",
            context=context,
            evidence=(PublicEvidence(evidence_id="e1", text=PREMISE),),
            max_tokens=_SYNTHESIS_MAX_TOKENS,
        )
        assert completion.text.strip(), (
            f"empty synthesis at max_tokens={_SYNTHESIS_MAX_TOKENS}"
        )
        assert not completion.truncated
