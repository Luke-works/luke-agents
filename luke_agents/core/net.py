"""Resolve the real client IP from ``X-Forwarded-For``, honoring trusted reverse-proxy hops.

``X-Forwarded-For`` is a comma list where each proxy APPENDS the peer it received the request from,
so the **leftmost** entry is client-authored and spoofable — a client that sends
``X-Forwarded-For: 1.2.3.4`` gets that prepended. Keying the rate limiter off the leftmost entry
therefore lets a caller mint a fresh budget per request by rotating a fake value, defeating the
whole AI-spend guard.

With ``N`` trusted proxies in front of the app, the real client is the ``N``-th entry counted from
the RIGHT. For this fleet that was **verified empirically to be 2** — Cloudflare sits in front of
Render, and each appends one entry (see ``AGENTS_TRUSTED_PROXY_HOPS``). ``0`` keeps the legacy
leftmost behavior for a single-hop / local dev run.
"""
from __future__ import annotations

import os

from fastapi import Request

# Default 2 = Cloudflare + Render (verified against the deployed service). Override per deployment.
_DEFAULT_TRUSTED_HOPS = 2


def _trusted_hops() -> int:
    raw = os.getenv("AGENTS_TRUSTED_PROXY_HOPS", "").strip()
    if not raw:
        return _DEFAULT_TRUSTED_HOPS
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_TRUSTED_HOPS


def client_ip(request: Request, trusted_hops: int | None = None) -> str:
    """The real client IP for rate-limiting / audit keys — spoof-resistant.

    Counts ``trusted_hops`` entries from the RIGHT of ``X-Forwarded-For`` (default from
    ``AGENTS_TRUSTED_PROXY_HOPS``, else 2). Falls back to the direct peer when no XFF is present
    (local/direct dev), and to ``"anon"`` if even that is missing — so it never raises.
    """
    n = trusted_hops if trusted_hops is not None else _trusted_hops()
    xff = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in xff.split(",") if h.strip()]
    if hops:
        if n <= 0:
            return hops[0]  # legacy leftmost (single-hop / local)
        idx = max(0, len(hops) - n)
        return hops[min(idx, len(hops) - 1)]
    return request.client.host if request.client else "anon"
