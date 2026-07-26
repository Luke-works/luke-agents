"""Prometheus metrics for the agent fleet (#22).

Exposes request volume + latency (with status, so rate-limit 429s and LLM 502s are visible),
and surfaces the transcript-durability counters (written/dropped/retried) at scrape time. Served
at the open `GET /metrics` endpoint in the Prometheus text format.

A dedicated registry keeps this self-contained (no clashes with the default global registry when
tests build many apps). Instrumentation is a pure-ASGI middleware, so it never buffers a streaming
response (the fine-tune export streams).
"""
from __future__ import annotations

import time

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import CounterMetricFamily

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS = Counter(
    "agents_requests_total", "HTTP requests handled, by method/route/status.",
    ["method", "route", "status"], registry=REGISTRY,
)
LATENCY = Histogram(
    "agents_request_latency_seconds", "Request handling latency in seconds, by route.",
    ["route"], registry=REGISTRY,
)


class _TranscriptCollector:
    """Surface the transcript write counters (from core.transcripts) at scrape time, so durability
    loss is visible in Prometheus without a second bookkeeping path."""

    def collect(self):
        try:
            from . import transcripts
            m = transcripts.metrics()
        except Exception:  # noqa: BLE001 - metrics must never break a scrape
            m = {}
        fam = CounterMetricFamily(
            "agents_transcript_writes", "Transcript write outcomes.", labels=["outcome"])
        fam.add_metric(["written"], m.get("turns_written", 0))
        fam.add_metric(["dropped"], m.get("turns_dropped", 0))
        fam.add_metric(["retried"], m.get("write_retries", 0))
        fam.add_metric(["feedback_dropped"], m.get("feedback_dropped", 0))
        yield fam


REGISTRY.register(_TranscriptCollector())


def render() -> bytes:
    return generate_latest(REGISTRY)


class MetricsMiddleware:
    """Pure-ASGI: count each request and observe its latency (status captured from the response
    start). Non-buffering, so streaming responses are unaffected."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        start = time.perf_counter()
        status_holder = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Label by the MATCHED route TEMPLATE (e.g. "/v1/agents/{slug}/chat"), not the raw path.
            # Starlette sets scope["route"] during routing; using the raw path would let an attacker
            # mint an unbounded number of label series (one per URL hit, incl. 404s) — a scrape/mem DoS.
            matched = scope.get("route")
            route = getattr(matched, "path", None) or "unmatched"
            method = scope.get("method", "GET")
            REQUESTS.labels(method, route, str(status_holder["code"])).inc()
            LATENCY.labels(route).observe(time.perf_counter() - start)
