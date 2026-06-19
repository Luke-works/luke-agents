"""Observability: per-request correlation IDs + structured (JSON) logging (#21).

A request carries an ``X-Correlation-Id`` (accepted if safe, else generated). The
middleware stores it in a context variable for the duration of the request, echoes
it on the response, and the logging setup tags every log line with it — so a
request is traceable across its log output (and lines up with the same header the
Java services use).
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar

HEADER = "X-Correlation-Id"

# Default "-" so logs emitted OUTSIDE a request (startup/background) still format.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")

_SAFE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def sanitize_correlation_id(candidate: str | None) -> str:
    """Accept a short, safe client id (no log-injection / unbounded values); else generate one."""
    if candidate is not None and _SAFE.match(candidate.strip()):
        return candidate.strip()
    return uuid.uuid4().hex


class CorrelationIdMiddleware:
    """Pure-ASGI middleware: set the request's correlation id in the context var (in
    the SAME context as the downstream app, so endpoints/logs see it — unlike
    BaseHTTPMiddleware, which runs the app in a detached task) and echo it on the
    response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        inbound = None
        for key, value in scope.get("headers", []):
            if key.decode("latin-1").lower() == "x-correlation-id":
                inbound = value.decode("latin-1")
                break
        cid = sanitize_correlation_id(inbound)
        token = correlation_id_var.set(cid)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((HEADER.encode("latin-1"), cid.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            correlation_id_var.reset(token)


class CorrelationIdFilter(logging.Filter):
    """Injects the current correlation id onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get()
        return True


class JsonLogFormatter(logging.Formatter):
    """One JSON object per log line, including the correlation id."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Route root logging through a single JSON handler tagged with the correlation id.
    Idempotent — safe to call once per process from the app factory."""
    global _configured
    if _configured:
        return
    _configured = True
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    handler.addFilter(CorrelationIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
