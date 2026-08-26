"""Centrally owned safety policy.

Safety is cross-cutting. Thresholds, risk classification, and the rules about
what a response must carry live here — not scattered across guardrails,
council, and agent modules.
"""
from __future__ import annotations

from mao.safety.policy import RiskLevel, SafetyPolicy, get_policy

__all__ = ["RiskLevel", "SafetyPolicy", "get_policy"]
