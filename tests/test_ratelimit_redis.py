"""#27: the Redis-backed limiter enforces a GLOBAL cap across workers/instances, and
the factory falls back to in-memory when REDIS_URL is unset (default-lenient dev mode).
"""
import pytest

from luke_agents.core import ratelimit
from luke_agents.core.ratelimit import (
    InMemoryRateLimiter,
    RedisRateLimiter,
    _build_limiter,
)


def test_factory_uses_in_memory_without_redis_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(_build_limiter(), InMemoryRateLimiter)


def test_factory_falls_back_when_redis_unreachable(monkeypatch):
    # REDIS_URL set but nothing listening → must not crash; degrade to in-memory.
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6390/0")
    assert isinstance(_build_limiter(), InMemoryRateLimiter)


def test_redis_limiter_caps_globally_across_instances():
    fakeredis = pytest.importorskip("fakeredis")
    # One shared server, two clients = two workers sharing the same Redis.
    server = fakeredis.FakeServer()
    worker_a = RedisRateLimiter(client=fakeredis.FakeStrictRedis(server=server))
    worker_b = RedisRateLimiter(client=fakeredis.FakeStrictRedis(server=server))

    key = "form:t:acme:ip:1.2.3.4"
    limit, window = 3, 3600

    # 3 allowed total across BOTH workers (not 3 per worker).
    assert worker_a.check_and_record(key, limit, window)[0] is True
    assert worker_b.check_and_record(key, limit, window)[0] is True
    assert worker_a.check_and_record(key, limit, window)[0] is True

    allowed, retry = worker_b.check_and_record(key, limit, window)
    assert allowed is False, "4th call must be denied — the cap is global, not per-worker"
    assert retry >= 1

    # A different key has its own budget.
    assert worker_a.check_and_record("other:key", limit, window)[0] is True


def test_redis_limiter_window_rollover():
    fakeredis = pytest.importorskip("fakeredis")
    rl = RedisRateLimiter(client=fakeredis.FakeStrictRedis())
    key = "k"
    # Fill the budget in a 1-second window...
    assert rl.check_and_record(key, 2, 1)[0] is True
    assert rl.check_and_record(key, 2, 1)[0] is True
    assert rl.check_and_record(key, 2, 1)[0] is False
    # ...wait it out; the old entries age out of the window and budget is restored.
    import time
    time.sleep(1.2)
    assert rl.check_and_record(key, 2, 1)[0] is True
