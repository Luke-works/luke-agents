"""Shared, default-lenient API-key gate for the agent endpoints (#32).

A single source of truth so the check can't drift between agents (the way the
inline copies in form_agent/email_agent could). Applied uniformly as a
router-level FastAPI dependency in ``server.build_app`` (``include_router(...,
dependencies=[Depends(require_api_key)])``) so EVERY current and FUTURE agent
route is covered by default — a new endpoint is gated automatically, not left
open until someone remembers to add the call.

Default-lenient: when ``AGENTS_API_KEY`` is unset (local dev, and the current
browser-direct flow), this is a no-op. Set ``AGENTS_API_KEY`` and have callers
send a matching ``X-Agents-Key`` header to fully close the unauthenticated
surface once traffic routes server-side. ``/health`` and ``/`` are declared at
the app level (not inside an agent router), so they intentionally stay open.
"""
from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request


def require_api_key(request: Request) -> None:
    """Reject the call with 401 when ``AGENTS_API_KEY`` is configured and the
    request does not present a matching ``X-Agents-Key`` header.

    Works both as a direct call ``require_api_key(req)`` and as a FastAPI
    dependency ``Depends(require_api_key)`` (FastAPI injects the ``Request``).
    """
    expected = os.getenv("AGENTS_API_KEY", "").strip()
    if not expected:
        return  # unconfigured → open (preserves the browser-direct dev flow)
    provided = request.headers.get("x-agents-key", "")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Valid API key required")
