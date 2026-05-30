"""Structured logging with per-request trace ID propagation."""
from __future__ import annotations

import contextvars
import json
import logging
import os
import time
from typing import Any

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def set_trace_id(trace_id: str) -> None:
    """Set the trace_id for the current async task / thread context."""
    _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """Return the current trace_id (empty string if not set)."""
    return _trace_id_var.get()


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    # Keys that are part of the standard LogRecord and should not be re-emitted as extras
    _STANDARD_KEYS = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        tid = _trace_id_var.get("")
        if tid:
            payload["trace_id"] = tid

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Copy any extra fields attached via logger.info(..., extra={"key": val})
        for key in vars(record):
            if key not in self._STANDARD_KEYS:
                payload[key] = getattr(record, key)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool | None = None) -> None:
    """Configure root logger with JSON or human-readable format.

    Args:
        level:     Log level string (DEBUG/INFO/WARNING/ERROR).
        json_logs: Force JSON mode. Defaults to True when LOG_FORMAT=json env var is set,
                   or when running in a container (detected via /.dockerenv existence).
    """
    if json_logs is None:
        json_logs = (
            os.getenv("LOG_FORMAT", "").lower() == "json"
            or os.path.exists("/.dockerenv")
        )

    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        ))

    root = logging.getLogger()
    # Remove only existing StreamHandlers to avoid disrupting non-stream integrations
    # (Sentry, OpenTelemetry, etc.) that may have registered other handler types.
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)]
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
