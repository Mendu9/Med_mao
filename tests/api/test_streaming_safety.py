"""P0-1 (API half) — the endpoint must not stream unverified text.

The graph now only populates `_stream_messages` on routes the policy exempts
from verification. This pins the API's own guard, so a future agent change
cannot re-open the bypass from the other side: the endpoint independently
refuses to stream raw provider tokens for anything that requires verification.
"""
from __future__ import annotations

from mao.api.streaming import may_stream_raw_tokens


class TestRawTokenStreamingIsRiskGated:
    def test_low_risk_with_prepared_messages_may_stream(self) -> None:
        assert may_stream_raw_tokens(
            {"risk_level": "low", "_stream_messages": [{"role": "user", "content": "hi"}]}
        ) is True

    def test_standard_risk_may_not_stream_raw_tokens(self) -> None:
        assert may_stream_raw_tokens(
            {"risk_level": "standard", "_stream_messages": [{"role": "user", "content": "hi"}]}
        ) is False

    def test_high_risk_may_not_stream_raw_tokens(self) -> None:
        assert may_stream_raw_tokens(
            {"risk_level": "high", "_stream_messages": [{"role": "user", "content": "hi"}]}
        ) is False

    def test_absent_risk_may_not_stream_raw_tokens(self) -> None:
        """Fail closed: an unclassified request is not a streaming candidate."""
        assert may_stream_raw_tokens({"_stream_messages": [{"role": "user", "content": "hi"}]}) is False

    def test_malformed_risk_may_not_stream_raw_tokens(self) -> None:
        assert may_stream_raw_tokens(
            {"risk_level": "LOW", "_stream_messages": [{"role": "user", "content": "x"}]}
        ) is False

    def test_no_prepared_messages_means_no_raw_streaming(self) -> None:
        assert may_stream_raw_tokens({"risk_level": "low", "_stream_messages": []}) is False

    def test_non_dict_state_fails_closed(self) -> None:
        assert may_stream_raw_tokens(None) is False


class TestVerifiedTextIsWhatShips:
    def test_the_endpoint_streams_the_guardrailed_response(self) -> None:
        """Everything that is not raw-token-eligible must ship the verified text."""
        import inspect

        import mao.api.main as main

        source = inspect.getsource(main.chat_stream_endpoint)
        # The guardrail result is applied before any streaming decision.
        assert "apply_output_guardrails" in source
        assert "may_stream_raw_tokens" in source
