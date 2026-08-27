"""Regression tests for the adversarial review findings on the routing work.

Each test pins one finding so it cannot silently return.
"""
from __future__ import annotations

import pytest

from mao.safety.policy import ATTACHMENT_KEYS, RiskLevel, get_policy, has_attachment


class TestC1VerificationCannotFailOpen:
    """C1 — a safety control that cannot load must stop the system, not be skipped.

    `except ImportError: return state` could not distinguish "not integrated yet"
    from "installed but its own imports blew up" (e.g. torch missing on an HF CPU
    Space). Either way every answer flowed to the user unverified behind one log line.
    """

    def test_graph_imports_the_verification_node_at_module_scope(self) -> None:
        import ast
        import inspect

        import mao.graph as graph

        tree = ast.parse(inspect.getsource(graph))
        module_level = {
            n.module
            for n in tree.body
            if isinstance(n, ast.ImportFrom) and n.module
        }
        assert "mao.safety.verification" in module_level

    def test_graph_does_not_swallow_a_verification_import_error(self) -> None:
        import inspect

        import mao.graph as graph

        source = inspect.getsource(graph)
        assert "except ImportError" not in source

    def test_the_wired_node_is_the_real_implementation(self) -> None:
        from mao.graph import build_graph
        from mao.safety import verification

        compiled = build_graph()
        assert "verification" in compiled.nodes
        assert verification.verification_node is not None


class TestH1AttachmentCoverage:
    """H1 — patient audio was classified STANDARD because the key list omitted it."""

    def test_attachment_keys_cover_audio(self) -> None:
        assert "audio_b64" in ATTACHMENT_KEYS
        assert "audio_path" in ATTACHMENT_KEYS

    def test_attachment_keys_cover_images_and_reports(self) -> None:
        for key in ("image_b64", "image_url", "report_b64", "report_path"):
            assert key in ATTACHMENT_KEYS

    @pytest.mark.parametrize("key", sorted(ATTACHMENT_KEYS))
    def test_any_attachment_makes_the_request_high_risk(self, key: str) -> None:
        assert get_policy().risk_for("graphrag", has_attachment=True) is RiskLevel.HIGH
        assert has_attachment({key: "payload"}) is True

    def test_audio_attachment_is_high_risk_through_the_risk_gate(self) -> None:
        from mao.graph import risk_gate_node

        out = risk_gate_node({"intent": "graphrag", "metadata": {"audio_b64": "voice"}})
        assert out["risk_level"] == RiskLevel.HIGH.value

    def test_no_attachment_is_not_high_risk(self) -> None:
        assert has_attachment({}) is False
        assert has_attachment({"unrelated": "x"}) is False

    def test_empty_attachment_value_does_not_count(self) -> None:
        assert has_attachment({"image_b64": ""}) is False

    def test_has_attachment_tolerates_a_non_dict(self) -> None:
        assert has_attachment(None) is False

    def test_the_constant_has_one_definition(self) -> None:
        """The key list must not be re-listed per module."""
        from mao.agents import multimodal_agent, router
        from mao import graph

        for module in (graph, router, multimodal_agent):
            import inspect

            source = inspect.getsource(module)
            assert '"image_b64", "image_url"' not in source, (
                f"{module.__name__} re-lists attachment keys instead of importing them"
            )


class TestH2ChitchatCannotDiscardAnAttachment:
    """H2 — an MRI plus the word "ok" was dispatched to a canned greeting and the
    scan was silently discarded, with no error and no log of the drop."""

    def test_gate_does_not_claim_chitchat_when_an_attachment_is_present(self) -> None:
        from mao.graph import chitchat_gate_node

        out = chitchat_gate_node(
            {"user_query": "ok", "metadata": {"image_b64": "<mri>"}}
        )
        assert out.get("intent") != "chitchat"

    @pytest.mark.parametrize("word", ["ok", "no", "yes", "good", "sure", "thanks"])
    def test_short_replies_with_a_scan_are_not_chitchat(self, word: str) -> None:
        """These are plausible answers to a clinical follow-up, not greetings."""
        from mao.graph import chitchat_gate_node

        out = chitchat_gate_node(
            {"user_query": word, "metadata": {"report_b64": "<report>"}}
        )
        assert out.get("intent") != "chitchat"

    def test_plain_chitchat_still_short_circuits(self) -> None:
        from mao.graph import chitchat_gate_node

        assert chitchat_gate_node({"user_query": "hello", "metadata": {}})["intent"] == "chitchat"

    def test_an_attached_request_reaches_a_clinical_capable_route(self) -> None:
        from mao.agents.router import route_to_agent
        from mao.graph import chitchat_gate_node

        state = chitchat_gate_node({"user_query": "ok", "metadata": {"image_b64": "<mri>"}})
        assert route_to_agent(state) != "chitchat_node"


class TestM2CouncilRoutingDefaultsClosed:
    """M2 — a malformed verdict routed the response to the user."""

    def test_missing_passed_key_routes_to_blocked(self) -> None:
        from mao.graph import _route_after_council

        assert _route_after_council({"council_verdict": {}}) == "blocked"

    def test_absent_verdict_routes_to_blocked(self) -> None:
        from mao.graph import _route_after_council

        assert _route_after_council({}) == "blocked"

    def test_non_dict_verdict_routes_to_blocked(self) -> None:
        from mao.graph import _route_after_council

        assert _route_after_council({"council_verdict": None}) == "blocked"

    def test_truthy_non_true_value_does_not_pass(self) -> None:
        from mao.graph import _route_after_council

        assert _route_after_council({"council_verdict": {"passed": "yes"}}) == "blocked"

    def test_an_explicit_pass_still_proceeds(self) -> None:
        from mao.graph import _route_after_council

        assert _route_after_council({"council_verdict": {"passed": True}}) == "senior_supervisor"


class TestM5RouterFailureDoesNotDeclassify:
    """M5 — a router outage returned graphrag, downgrading clinical requests to
    STANDARD and dropping the mandatory disclaimer with no user-visible signal."""

    def test_provider_outage_marks_the_classification_as_failed(self) -> None:
        """Patch the provider call itself, not the classifier, so the real
        fallback path runs."""
        from unittest.mock import patch

        from mao.agents.router import _classify

        with patch("mao.core.llm.chat", side_effect=RuntimeError("provider down")):
            intent, ok = _classify("sys", "user")
        assert ok is False

    def test_an_unrecognised_label_also_counts_as_failure(self) -> None:
        from unittest.mock import patch

        from mao.agents.router import _classify

        with patch("mao.core.llm.chat", return_value="banana"):
            intent, ok = _classify("sys", "user")
        assert ok is False

    def test_a_recognised_label_counts_as_success(self) -> None:
        from unittest.mock import patch

        from mao.agents.router import _classify

        with patch("mao.core.llm.chat", return_value="clinical"):
            intent, ok = _classify("sys", "user")
        assert (intent, ok) == ("clinical", True)

    def test_router_node_records_the_failure_on_state(self, monkeypatch) -> None:
        from mao.agents import router

        monkeypatch.setattr(router, "_classify", lambda *a, **k: ("graphrag", False))
        monkeypatch.setattr(router, "search_memories", lambda *a, **k: "")
        out = router.router_node(
            {"user_query": "what stage is this patient?", "user_id": "u", "metadata": {}}
        )
        assert out.get("router_failed") is True

    def test_a_failed_router_escalates_risk_to_high(self) -> None:
        from mao.graph import risk_gate_node

        out = risk_gate_node({"intent": "graphrag", "metadata": {}, "router_failed": True})
        assert out["risk_level"] == RiskLevel.HIGH.value

    def test_a_healthy_router_does_not_escalate(self) -> None:
        from mao.graph import risk_gate_node

        out = risk_gate_node({"intent": "graphrag", "metadata": {}})
        assert out["risk_level"] == RiskLevel.STANDARD.value

    def test_escalated_risk_requires_the_clinical_disclaimer(self) -> None:
        assert get_policy().requires_clinical_disclaimer(RiskLevel.HIGH) is True
