"""Typed tool capabilities.

Importing this package registers the built-in tools, so `registry()` is
populated for any consumer — the ReAct agent, a future capability router, or a
trace that needs to resolve a `tool_id`.
"""
from __future__ import annotations

from mao.tools.builtin import register_builtin_tools
from mao.tools.registry import (
    ReadOrWrite,
    ToolRegistry,
    ToolSpec,
    TrustTier,
    get_tool,
    registry,
)

register_builtin_tools()

__all__ = [
    "ReadOrWrite",
    "ToolRegistry",
    "ToolSpec",
    "TrustTier",
    "get_tool",
    "registry",
]
