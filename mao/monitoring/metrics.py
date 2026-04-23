"""
mao/monitoring/metrics.py
--------------------------
Prometheus metrics for MAO.

Imported by mao/api/main.py which mounts /metrics via make_asgi_app().

Metrics exposed:
  mao_requests_total          Counter   — total requests by agent + intent
  mao_latency_seconds         Histogram — request latency by agent
  mao_faithfulness_score      Gauge     — latest RAGAS faithfulness by agent
  mao_hallucination_flag      Gauge     — 1.0 if last response faithfulness < 0.7
  mao_reranker_top_score      Histogram — top reranker score per graphrag/clinical request
  mao_active_requests         Gauge     — in-flight requests

Usage (from api/main.py):
    from mao.monitoring.metrics import record_request, record_reranker_score

Grafana dashboard panels:
  - Request rate:       rate(mao_requests_total[5m])
  - P95 latency:        histogram_quantile(0.95, mao_latency_seconds_bucket)
  - Faithfulness trend: mao_faithfulness_score
  - Hallucination rate: avg_over_time(mao_hallucination_flag[1h])

Requires: pip install prometheus-client
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    # Total requests labelled by agent and intent
    request_counter = Counter(
        "mao_requests_total",
        "Total number of MAO chat requests",
        ["agent", "intent"],
    )

    # Request latency distribution
    latency_histogram = Histogram(
        "mao_latency_seconds",
        "End-to-end request latency in seconds",
        ["agent"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
    )

    # Latest RAGAS faithfulness score per agent (set after async scoring completes)
    faithfulness_gauge = Gauge(
        "mao_faithfulness_score",
        "Latest RAGAS faithfulness score (0-1, higher is better)",
        ["agent"],
    )

    # 1.0 if last response was below faithfulness threshold (potential hallucination)
    hallucination_rate_gauge = Gauge(
        "mao_hallucination_flag",
        "1.0 if last response faithfulness < 0.7 (potential hallucination)",
        ["agent"],
    )

    # Top reranker score for each retrieval request
    reranker_score_histogram = Histogram(
        "mao_reranker_top_score",
        "Top reranker score per retrieval request",
        ["agent"],
        buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
    )

    # In-flight requests
    active_requests_gauge = Gauge(
        "mao_active_requests",
        "Number of in-flight /chat requests",
    )

    _PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus metrics registered.")

except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus-client not installed — metrics disabled. "
        "Run: pip install prometheus-client"
    )

    # Stub objects so imports don't fail elsewhere
    class _Stub:
        def labels(self, **_): return self
        def inc(self, *_): pass
        def observe(self, *_): pass
        def set(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass

    request_counter          = _Stub()
    latency_histogram        = _Stub()
    faithfulness_gauge       = _Stub()
    hallucination_rate_gauge = _Stub()
    reranker_score_histogram = _Stub()
    active_requests_gauge    = _Stub()


# ---------------------------------------------------------------------------
# Convenience helpers called from api/main.py
# ---------------------------------------------------------------------------

def record_request(agent: str, intent: str, latency_seconds: float) -> None:
    """Increment request counter and record latency. Called after each /chat."""
    try:
        request_counter.labels(agent=agent, intent=intent).inc()
        latency_histogram.labels(agent=agent).observe(latency_seconds)
    except Exception:  # noqa: BLE001
        pass


def record_reranker_score(agent: str, top_score: float) -> None:
    """Record the top reranker score for a retrieval request."""
    try:
        reranker_score_histogram.labels(agent=agent).observe(top_score)
    except Exception:  # noqa: BLE001
        pass
