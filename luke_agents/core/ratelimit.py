"""Per-caller sliding-window rate limit, shared by every agent.

Caps each caller to RATE_LIMIT_MAX actions per RATE_LIMIT_WINDOW seconds (default
200 per 3 hours) so a single user — or a runaway script — can't drive AI spend
out of hand. Agents namespace their keys (e.g. "form:t:acme:ip:1.2.3.4") so each
agent gets its own independent budget per caller.

Backend is pluggable (#27):
  * REDIS_URL set  → RedisRateLimiter: a sliding-window log in a Redis sorted set,
    so the window is GLOBAL across every uvicorn worker and instance and survives
    restarts/redeploys. This is required for any multi-worker / HA / autoscaled run —
    otherwise the budget is per-worker and the effective cap is multiplied.
  * REDIS_URL unset (or Redis unreachable at boot) → InMemoryRateLimiter: the
    process-local fallback, correct only for a single always-on single-worker
    instance. Documented dev/single-worker mode.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque

log = logging.getLogger(__name__)

# Module-level config (read at call time so tests can monkeypatch them).
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "200"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", str(3 * 60 * 60)))


class RateLimiter(ABC):
    """Records an action for a key and reports whether it is within budget."""

    @abstractmethod
    def check_and_record(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds). retry_after is how long until the
        oldest action in the window ages out (only meaningful when not allowed)."""
        raise NotImplementedError


class InMemoryRateLimiter(RateLimiter):
    """Process-local sliding window. Correct ONLY for one instance with one worker."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._calls = 0

    def _sweep(self, now: float, window: int) -> None:
        cutoff = now - window
        stale = [k for k, dq in self._hits.items() if not dq or dq[-1] < cutoff]
        for k in stale:
            del self._hits[k]

    def check_and_record(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - window
        with self._lock:
            dq = self._hits[key]
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= limit:
                retry_after = int(dq[0] + window - now) + 1
                return False, max(retry_after, 1)
            dq.append(now)
            self._calls += 1
            if self._calls % 500 == 0:
                self._sweep(now, window)
            return True, 0


class RedisRateLimiter(RateLimiter):
    """Global sliding-window log in a Redis sorted set (score = timestamp).

    Atomic per call: ZREMRANGEBYSCORE (evict) + ZADD (record) + ZCARD (count) run in
    one transactional pipeline; Redis is single-threaded so concurrent workers serialize,
    giving an exact global cap. PEXPIRE keeps idle keys from leaking.
    """

    def __init__(self, url: str | None = None, client=None) -> None:
        if client is not None:
            self._client = client
        else:
            import redis  # imported lazily so the dep is only needed when REDIS_URL is set
            self._client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
            self._client.ping()  # fail fast at construction → factory falls back to in-memory

    def check_and_record(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = time.time()
        member = f"{now:.6f}-{uuid.uuid4().hex}"
        pipe = self._client.pipeline()  # transactional (MULTI/EXEC) by default
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zadd(key, {member: now})
        pipe.zcard(key)
        pipe.pexpire(key, int(window * 1000) + 1000)
        count = pipe.execute()[2]
        if count <= limit:
            return True, 0
        # Over budget: undo our own record so the set doesn't stay inflated, then report
        # how long until the oldest in-window action ages out.
        self._client.zrem(key, member)
        oldest = self._client.zrange(key, 0, 0, withscores=True)
        retry_after = 1
        if oldest:
            retry_after = max(1, int(oldest[0][1] + window - now) + 1)
        return False, retry_after


def _build_limiter() -> RateLimiter:
    url = os.getenv("REDIS_URL", "").strip()
    if url:
        try:
            limiter = RedisRateLimiter(url)
            log.info("rate limiter: Redis (global, multi-worker safe)")
            return limiter
        except Exception as e:  # noqa: BLE001 - any connection/config error → safe fallback
            log.warning("rate limiter: REDIS_URL set but Redis is unavailable (%s) — "
                        "falling back to in-memory (per-worker budget). Fix Redis for HA.", e)
    else:
        log.info("rate limiter: in-memory (single-worker only; set REDIS_URL for multi-worker)")
    return InMemoryRateLimiter()


_limiter: RateLimiter = _build_limiter()


def reset() -> None:
    """Rebuild the limiter backend, clearing in-memory window state. Primarily for
    tests (and a safe operational reset). For Redis this reconnects but does NOT flush
    the shared store — the global window is intentionally durable."""
    global _limiter
    _limiter = _build_limiter()


def check_and_record(key: str) -> tuple[bool, int]:
    """Record an action for `key` and report whether it's allowed.

    Returns (allowed, retry_after_seconds). Reads the module-level limit/window at call
    time and delegates to the selected backend.
    """
    return _limiter.check_and_record(key, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW)


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
