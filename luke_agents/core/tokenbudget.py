"""Per-tenant DAILY token budget — a spend ceiling layered on top of the per-caller rate
limit (D5).

The rate limiter (``ratelimit.py``) caps *how many actions* a caller makes; this caps *how
many LLM tokens a TENANT consumes per UTC day*, which is the fleet's real cost driver
(surfaced by D6's ``agents_llm_tokens_total`` / ``llm.last_usage()``). A single tenant spread
across many IPs and user_ids can stay under every per-caller limit yet still run up unbounded
token spend — this closes that hole with one number the operator controls.

DEFAULT-LENIENT: OFF unless a cap is configured. Unset / 0 / invalid = no cap, so dev/qa keep
working with zero config. Arm a flat cap for every tenant with e.g.
``AGENTS_TENANT_DAILY_TOKEN_CAP=1000000`` (~$9/mo worst-case per tenant on the default Groq model).

TIER-AWARE: a tenant's cap is sized by its commercial plan. When ``AGENTS_TOKEN_CAP_<TIER>`` is set
(``AGENTS_TOKEN_CAP_FREE`` / ``_PRO`` / ``_BUSINESS`` / ``_ENTERPRISE``), a request carrying that tier
(the ``X-Tenant-Tier`` header, resolved by ``tenancy.resolve_tier``) is capped at that number; a tier
with no configured cap — or an absent/unknown tier — falls back to the flat cap. So a bigger plan
gets a bigger AI budget, and a missing tier can neither block a request nor silently widen its budget.
The day's usage is recorded whenever ANY cap is configured; only the ceiling checked at ``enforce()``
depends on the tier.

Backend mirrors ``ratelimit.py``:
  * ``REDIS_URL`` set   → a per-tenant, per-UTC-day counter in Redis (INCRBY + ~2-day EXPIRE),
    so the budget is GLOBAL across every worker/instance and resets naturally each day.
  * ``REDIS_URL`` unset → a process-local in-memory counter (correct for single-worker dev only;
    the same limitation the rate limiter has without Redis).

Flow (check-then-charge). ``enforce(tenant)`` at the top of a paid endpoint binds the tenant to
the request and rejects it if it has already reached today's cap; every ``llm.generate()`` then
records its actual usage via ``record_current()`` (called from ``llm._note_usage``, so multi-call
paths like document intake are SUMMED, not just the last turn). The turn that crosses the line
still completes; the tenant's *next* turn that day is blocked until midnight UTC.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextvars import ContextVar

log = logging.getLogger(__name__)

# Request-scoped tenant, set by enforce() so record_current() (invoked from the LLM layer, which
# has no tenant of its own) can attribute usage without threading the tenant through every call.
_CURRENT_TENANT: ContextVar[str | None] = ContextVar("_budget_tenant", default=None)

_SECONDS_PER_DAY = 24 * 60 * 60


# The commercial plan tiers a per-tier cap can be set for (mirror of core-engine PlanCatalog ids).
_TIERS = ("FREE", "PRO", "BUSINESS", "ENTERPRISE")


def _int_env(name: str) -> int:
    """Read a non-negative int env knob; unset / 0 / invalid → 0. Read at call time so an operator
    (or a test) can change it without a reimport."""
    try:
        return int(os.getenv(name, "0"))
    except ValueError:
        return 0


def _flat_cap() -> int:
    """The tier-agnostic daily cap (``AGENTS_TENANT_DAILY_TOKEN_CAP``); <= 0 means unset."""
    return _int_env("AGENTS_TENANT_DAILY_TOKEN_CAP")


def _tier_cap(tier: str | None) -> int:
    """The daily cap configured for a specific tier (``AGENTS_TOKEN_CAP_<TIER>``), or 0 if unset."""
    if not tier:
        return 0
    return _int_env(f"AGENTS_TOKEN_CAP_{tier.strip().upper()}")


def _cap_for(tier: str | None) -> int:
    """Today's cap for a request on `tier`: the tier-specific cap when configured (> 0), otherwise
    the flat cap. <= 0 means DISABLED for this request — the default-lenient outcome for an unknown
    tier when the flat cap is also unset, so a missing/garbage tier can never block or widen budget."""
    tc = _tier_cap(tier)
    return tc if tc > 0 else _flat_cap()


def _metering_on() -> bool:
    """True when ANY cap (flat or any tier) is configured. Gates whether we record usage at all, so
    a fully-unconfigured deployment does zero accounting work (and stays exactly as it was)."""
    return _flat_cap() > 0 or any(_int_env(f"AGENTS_TOKEN_CAP_{t}") > 0 for t in _TIERS)


def _utc_day(now: float) -> str:
    """UTC calendar day as YYYYMMDD — the natural, timezone-stable reset boundary."""
    return time.strftime("%Y%m%d", time.gmtime(now))


def _seconds_to_utc_midnight(now: float) -> int:
    return int(_SECONDS_PER_DAY - (now % _SECONDS_PER_DAY)) + 1


class TokenCounter(ABC):
    """Accumulates a tenant's token usage within a UTC day."""

    @abstractmethod
    def get(self, tenant: str, day: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def add(self, tenant: str, day: str, tokens: int) -> None:
        raise NotImplementedError


class InMemoryTokenCounter(TokenCounter):
    """Process-local daily counter. Correct ONLY for one instance with one worker."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self._lock = threading.Lock()

    def get(self, tenant: str, day: str) -> int:
        with self._lock:
            return self._counts.get((tenant, day), 0)

    def add(self, tenant: str, day: str, tokens: int) -> None:
        with self._lock:
            self._counts[(tenant, day)] += tokens
            # Drop this tenant's other days so yesterday's keys can't accumulate forever.
            stale = [k for k in self._counts if k[0] == tenant and k[1] != day]
            for k in stale:
                del self._counts[k]


class RedisTokenCounter(TokenCounter):
    """Global per-tenant daily counter in Redis: INCRBY the day key, EXPIRE it ~2 days out so it
    self-cleans. Redis is single-threaded, so concurrent workers serialize to an exact total."""

    def __init__(self, url: str | None = None, client=None) -> None:
        if client is not None:
            self._client = client
        else:
            import redis  # imported lazily so the dep is only needed when REDIS_URL is set

            self._client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
            self._client.ping()  # fail fast at construction → factory falls back to in-memory

    def _key(self, tenant: str, day: str) -> str:
        return f"agents:tokbudget:{tenant}:{day}"

    def get(self, tenant: str, day: str) -> int:
        v = self._client.get(self._key(tenant, day))
        return int(v) if v else 0

    def add(self, tenant: str, day: str, tokens: int) -> None:
        key = self._key(tenant, day)
        pipe = self._client.pipeline()
        pipe.incrby(key, tokens)
        pipe.expire(key, _SECONDS_PER_DAY * 2)
        pipe.execute()


def _build_counter() -> TokenCounter:
    url = os.getenv("REDIS_URL", "").strip()
    if url:
        try:
            counter = RedisTokenCounter(url)
            log.info("token budget: Redis (global, multi-worker safe)")
            return counter
        except Exception as e:  # noqa: BLE001 - any connection/config error → safe fallback
            log.warning("token budget: REDIS_URL set but Redis is unavailable (%s) — falling "
                        "back to in-memory (per-worker counts). Fix Redis for a global cap.", e)
    return InMemoryTokenCounter()


_counter: TokenCounter = _build_counter()


def reset() -> None:
    """Rebuild the backend (clears in-memory state) and clear the request-scoped tenant.
    For tests and a safe operational reset. For Redis this reconnects but does NOT flush the
    shared counters — the daily budget is intentionally durable and expires on its own."""
    global _counter
    _counter = _build_counter()
    _CURRENT_TENANT.set(None)


def current_usage(tenant: str) -> int:
    """Tokens recorded for `tenant` so far today (UTC)."""
    return _counter.get(tenant, _utc_day(time.time()))


def enforce(tenant: str, tier: str | None = None) -> None:
    """Bind `tenant` to this request (so its LLM usage is attributed) and raise HTTP 429 if it has
    already reached today's token cap. The cap is chosen by `tier` (``AGENTS_TOKEN_CAP_<TIER>``) when
    that tier is armed, else the flat cap — so a higher plan gets a bigger AI budget. No-op — beyond
    binding the tenant — when the resolved cap is disabled, so dev/qa (and any un-armed tier) are
    never blocked. Agents call this right after the rate-limit ``enforce``, passing the tier from
    ``resolve_tier(request)``."""
    _CURRENT_TENANT.set(tenant)
    cap = _cap_for(tier)
    if cap <= 0:
        return
    now = time.time()
    if _counter.get(tenant, _utc_day(now)) < cap:
        return
    from fastapi import HTTPException

    retry_after = _seconds_to_utc_midnight(now)
    hours = max(1, round(retry_after / 3600))
    raise HTTPException(
        status_code=429,
        detail=f"Your organization has reached its daily AI usage limit of {cap:,} tokens. "
        f"It resets at midnight UTC (in about {hours} hour{'s' if hours != 1 else ''}).",
        headers={"Retry-After": str(retry_after)},
    )


def record(tenant: str, tokens: int) -> None:
    """Add `tokens` to `tenant`'s usage for today (UTC). No-op when metering is entirely off (no flat
    or tier cap configured) or tokens <= 0 — when nothing is capped there's nothing to meter. The
    day's tally is tier-agnostic (just total tokens); the tier only decides the ceiling at enforce."""
    if tokens <= 0 or not _metering_on():
        return
    _counter.add(tenant, _utc_day(time.time()), int(tokens))


def record_current(tokens: int) -> None:
    """Record `tokens` against the request-scoped tenant set by ``enforce()``. Called from the LLM
    layer after EACH turn, so every ``generate()`` — including multi-call paths — is summed.
    Best-effort: never raises into the request."""
    try:
        tenant = _CURRENT_TENANT.get()
        if tenant:
            record(tenant, tokens)
    except Exception:  # noqa: BLE001 - accounting must never break a turn
        pass
