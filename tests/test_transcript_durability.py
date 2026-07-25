"""#41 / #30 / #28 — durable, observable transcript writes.

Covers: transient-error classification + bounded retry-with-backoff, the off-path write
queue (retry-then-success, drop-and-count on exhaustion), the shutdown flush, and the
production ephemeral-storage guard. No live DB or LLM needed — the store is faked.
"""
import pytest

from luke_agents.core import transcripts as T
from luke_agents.core.transcripts import (
    Feedback, JsonlStore, NullStore, PostgresStore, TurnRecord,
    _build_store, _is_transient, _with_retry, flush_pending, metrics,
    safe_record_feedback, safe_record_turn,
)


# Exceptions matched by class NAME (transcripts avoids importing psycopg2 in dev mode).
class PoolError(Exception):
    pass


class OperationalError(Exception):
    pass


def _rec(rid: str = "t1") -> TurnRecord:
    return TurnRecord(
        id=rid, agent="form", brain="test", model="m",
        messages=[{"role": "system", "content": "S"}], output={"ok": True},
    )


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """Fresh writer + zeroed counters + zero backoff for each test."""
    monkeypatch.setenv("AGENTS_TRANSCRIPT_RETRY_BASE_SECONDS", "0")
    monkeypatch.setattr(T, "_writer", None)
    monkeypatch.setattr(T, "_metrics",
                        {"turns_written": 0, "turns_dropped": 0, "write_retries": 0, "feedback_dropped": 0})
    yield


# --- transient classification (#41/#30) ---------------------------------------

def test_is_transient_matches_pool_and_connection_errors():
    assert _is_transient(PoolError("connection pool exhausted"))
    assert _is_transient(OperationalError("server closed the connection"))
    assert not _is_transient(ValueError("programming error"))


def test_is_transient_follows_cause_chain():
    try:
        try:
            raise PoolError("exhausted")
        except Exception as inner:
            raise RuntimeError("wrapped") from inner
    except Exception as outer:
        assert _is_transient(outer)


# --- bounded retry-with-backoff (#41) -----------------------------------------

def test_with_retry_retries_transient_then_succeeds():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise PoolError("pool exhausted")
        return "ok"

    assert _with_retry(fn, what="test") == "ok"
    assert calls["n"] == 3
    assert metrics()["write_retries"] == 2


def test_with_retry_gives_up_after_attempts(monkeypatch):
    monkeypatch.setenv("AGENTS_TRANSCRIPT_WRITE_RETRIES", "2")
    with pytest.raises(PoolError):
        _with_retry(lambda: (_ for _ in ()).throw(PoolError("always")), what="test")


def test_with_retry_does_not_retry_non_transient():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ValueError("bad")

    with pytest.raises(ValueError):
        _with_retry(fn, what="test")
    assert calls["n"] == 1  # no retry on a non-transient error


# --- the off-path writer queue (#41) ------------------------------------------

def test_writer_persists_after_transient_failure(monkeypatch):
    written = []

    class FlakyStore:
        def __init__(self):
            self.n = 0

        def record_turn(self, rec):
            self.n += 1
            if self.n < 2:
                raise PoolError("exhausted")
            written.append(rec.id)

    store = FlakyStore()  # ONE instance, so its failure counter persists across retries
    monkeypatch.setattr(T, "get_store", lambda: store)
    safe_record_turn(_rec("w1"))
    assert flush_pending(timeout=3.0) == 0           # drained on (simulated) shutdown
    assert written == ["w1"]                          # eventually written despite the blip
    m = metrics()
    assert m["turns_written"] == 1 and m["turns_dropped"] == 0 and m["write_retries"] >= 1


def test_writer_drops_and_counts_after_exhaustion(monkeypatch):
    monkeypatch.setenv("AGENTS_TRANSCRIPT_WRITE_RETRIES", "2")

    class DeadStore:
        def record_turn(self, rec):
            raise PoolError("always exhausted")

    monkeypatch.setattr(T, "get_store", lambda: DeadStore())
    safe_record_turn(_rec("d1"))
    flush_pending(timeout=3.0)
    m = metrics()
    assert m["turns_dropped"] == 1 and m["turns_written"] == 0  # loss is observable, not silent


def test_flush_pending_is_noop_when_never_used():
    assert flush_pending(0.1) == 0


# --- feedback retry (sync path) -----------------------------------------------

def test_feedback_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class Store:
        def record_feedback(self, turn_id, fb):
            calls["n"] += 1
            if calls["n"] < 2:
                raise OperationalError("connection reset")
            return True

    monkeypatch.setattr(T, "get_store", lambda: Store())
    assert safe_record_feedback("t1", Feedback(rating=1)) is True
    assert calls["n"] == 2


# --- production ephemeral-storage guard (#28) ---------------------------------

def test_prod_without_database_url_disables_recording(monkeypatch):
    monkeypatch.setenv("AGENTS_ENV", "production")
    monkeypatch.setenv("TRANSCRIPTS_ENABLED", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = _build_store()
    assert isinstance(store, NullStore)  # NOT ephemeral JSONL


def test_dev_without_database_url_uses_ephemeral_jsonl(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTS_ENV", raising=False)
    monkeypatch.setenv("TRANSCRIPTS_ENABLED", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TRANSCRIPTS_DIR", str(tmp_path))
    store = _build_store()
    assert isinstance(store, JsonlStore) and store.ephemeral is True


def test_database_url_selects_durable_postgres(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTS_ENABLED", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/db")
    store = _build_store()  # lazy pool: constructing the store does not connect
    assert isinstance(store, PostgresStore) and store.ephemeral is False


# --- configurable pool sizing (#30) -------------------------------------------

def test_pool_uses_env_sizing(monkeypatch):
    pytest.importorskip("psycopg2")
    import psycopg2.pool as pp

    captured = {}

    class FakePool:
        def __init__(self, mn, mx, dsn):
            captured.update(mn=mn, mx=mx, dsn=dsn)

    monkeypatch.setattr(pp, "ThreadedConnectionPool", FakePool)
    monkeypatch.setenv("AGENTS_DB_POOL_MIN", "2")
    monkeypatch.setenv("AGENTS_DB_POOL_MAX", "9")
    store = PostgresStore("postgresql://u:p@h/db", "luke_agents")
    store._pool_or_connect()
    assert captured["mn"] == 2 and captured["mx"] == 9
