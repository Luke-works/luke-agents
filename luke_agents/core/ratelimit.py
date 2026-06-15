"""Per-caller sliding-window rate limit, shared by every agent.

Caps each caller to RATE_LIMIT_MAX actions per RATE_LIMIT_WINDOW seconds (default
200 per 3 hours) so a single user — or a runaway script — can't drive AI spend
out of hand. Agents namespace their keys (e.g. "form:user:123") so each agent
gets its own independent budget per caller.

In-memory and process-local: correct for a single always-on instance with one
worker (the current Render Starter setup). If you scale to multiple workers /
instances, move this to a shared store (e.g. Redis) so the window is global.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "200"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", str(3 * 60 * 60)))

_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()
_calls = 0


def _sweep(now: float) -> None:
    """Drop keys with no recent activity so memory stays bounded. Caller holds
    the lock; runs occasionally, not every request."""
    cutoff = now - RATE_LIMIT_WINDOW
    stale = [k for k, dq in _hits.items() if not dq or dq[-1] < cutoff]
    for k in stale:
        del _hits[k]


def check_and_record(key: str) -> tuple[bool, int]:
    """Record an action for `key` and report whether it's allowed.

    Returns (allowed, retry_after_seconds). When not allowed, retry_after is how
    long until the oldest action in the window ages out.
    """
    global _calls
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    with _lock:
        dq = _hits[key]
        while dq and dq[0] < cutoff:
            dq.popleft()

        if len(dq) >= RATE_LIMIT_MAX:
            retry_after = int(dq[0] + RATE_LIMIT_WINDOW - now) + 1
            return False, max(retry_after, 1)

        dq.append(now)
        _calls += 1
        if _calls % 500 == 0:
            _sweep(now)
        return True, 0


def enforce(key: str) -> None:
    """Convenience wrapper: raise HTTP 429 (with a friendly message + Retry-After)
    when `key` is over budget. Agents call this at the top of a paid endpoint."""
    from fastapi import HTTPException

    allowed, retry_after = check_and_record(key)
    if allowed:
        return
    hours = RATE_LIMIT_WINDOW // 3600
    window = f"{hours} hours" if hours != 1 else "hour"
    mins = max(1, round(retry_after / 60))
    raise HTTPException(
        status_code=429,
        detail=f"You've reached the limit of {RATE_LIMIT_MAX} actions per {window}. "
        f"Please try again in about {mins} minute{'s' if mins != 1 else ''}.",
        headers={"Retry-After": str(retry_after)},
    )
