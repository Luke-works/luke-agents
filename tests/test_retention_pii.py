"""#29 — retention / PII / GDPR controls on transcripts.

Exercises the JsonlStore (dev backend) end-to-end for:
  - consent=false stores MINIMALLY (metadata kept, content dropped);
  - optional PII redaction before persistence;
  - right-to-erasure by user_id / session_id;
  - retention purge by created_at;
and the config/redaction helpers directly.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from luke_agents.core import transcripts as T
from luke_agents.core.transcripts import JsonlStore, TurnRecord, redact, retention_days


def _rec(**kw) -> TurnRecord:
    base = dict(
        id=kw.pop("id", "t1"), agent="form", brain="test", model="m",
        messages=[
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "email me at a@b.com"},
        ],
        output={"reply": "ok", "email": "a@b.com"},
        input_schema={"entities": {}, "root": []},
    )
    base.update(kw)
    return TurnRecord(**base)


def _read(store: JsonlStore) -> list[dict]:
    return [json.loads(l) for l in store.turns.read_text().splitlines() if l.strip()]


# --- consent gates STORAGE, not just export ------------------------------------

def test_consent_false_stores_minimally(tmp_path):
    store = JsonlStore(str(tmp_path))
    store.record_turn(_rec(consent=False))
    rows = _read(store)
    assert len(rows) == 1
    row = rows[0]
    # metadata retained
    assert row["consent"] is False and row["id"] == "t1" and row["agent"] == "form"
    # content dropped: no user/system text, no output, no input schema
    assert all(m["content"] == "" for m in row["messages"])
    assert row["output"] is None
    assert row["input_schema"] is None
    # prompt_hash still computed from the real system prompt
    assert row["prompt_hash"] is not None


def test_consent_true_keeps_content(tmp_path):
    store = JsonlStore(str(tmp_path))
    store.record_turn(_rec(consent=True))
    row = _read(store)[0]
    assert row["output"] == {"reply": "ok", "email": "a@b.com"}
    assert any("a@b.com" in m["content"] for m in row["messages"])


# --- redaction hook -------------------------------------------------------------

def test_redact_scrubs_pii():
    out = redact({"msg": "reach me at a@b.com or +1 (415) 555-1234, ssn 123-45-6789"})
    s = out["msg"]
    assert "a@b.com" not in s and "[redacted-email]" in s
    assert "555-1234" not in s and "[redacted-phone]" in s
    assert "123-45-6789" not in s and "[redacted-ssn]" in s


def test_redaction_applied_on_store_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTS_REDACT_PII", "true")
    store = JsonlStore(str(tmp_path))
    store.record_turn(_rec(consent=True))
    row = _read(store)[0]
    assert not any("a@b.com" in m["content"] for m in row["messages"])
    assert "a@b.com" not in json.dumps(row["output"])


def test_redaction_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTS_REDACT_PII", raising=False)
    store = JsonlStore(str(tmp_path))
    store.record_turn(_rec(consent=True))
    row = _read(store)[0]
    assert any("a@b.com" in m["content"] for m in row["messages"])


# --- right to erasure -----------------------------------------------------------

def test_delete_for_user(tmp_path):
    store = JsonlStore(str(tmp_path))
    store.record_turn(_rec(id="t1", user_id="u1"))
    store.record_turn(_rec(id="t2", user_id="u2"))
    store.record_turn(_rec(id="t3", user_id="u1"))
    n = store.delete_for_user(user_id="u1")
    assert n == 2
    ids = {r["id"] for r in _read(store)}
    assert ids == {"t2"}


def test_delete_for_session(tmp_path):
    store = JsonlStore(str(tmp_path))
    store.record_turn(_rec(id="t1", session_id="s1"))
    store.record_turn(_rec(id="t2", session_id="s2"))
    assert store.delete_for_user(session_id="s1") == 1
    assert {r["id"] for r in _read(store)} == {"t2"}


def test_delete_for_user_requires_an_identifier(tmp_path):
    store = JsonlStore(str(tmp_path))
    with pytest.raises(ValueError):
        store.delete_for_user()


def test_delete_for_user_also_drops_feedback(tmp_path):
    store = JsonlStore(str(tmp_path))
    store.record_turn(_rec(id="t1", user_id="u1"))
    store.record_feedback("t1", T.Feedback(rating=1))
    store.delete_for_user(user_id="u1")
    fb = [l for l in store.feedback.read_text().splitlines() if l.strip()]
    assert fb == []


# --- retention purge ------------------------------------------------------------

def test_purge_older_than(tmp_path):
    store = JsonlStore(str(tmp_path))
    old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    new = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    store.record_turn(_rec(id="old", created_at=old))
    store.record_turn(_rec(id="new", created_at=new))
    n = store.purge_older_than(30)
    assert n == 1
    assert {r["id"] for r in _read(store)} == {"new"}


def test_purge_zero_days_is_noop(tmp_path):
    store = JsonlStore(str(tmp_path))
    store.record_turn(_rec(id="a"))
    assert store.purge_older_than(0) == 0
    assert len(_read(store)) == 1


# --- config helper --------------------------------------------------------------

def test_retention_days_config(monkeypatch):
    monkeypatch.delenv("AGENTS_RETENTION_DAYS", raising=False)
    assert retention_days() is None
    monkeypatch.setenv("AGENTS_RETENTION_DAYS", "0")
    assert retention_days() is None
    monkeypatch.setenv("AGENTS_RETENTION_DAYS", "45")
    assert retention_days() == 45
    monkeypatch.setenv("AGENTS_RETENTION_DAYS", "notanint")
    assert retention_days() is None


# --- module-level wrappers ------------------------------------------------------

def test_module_purge_respects_env(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TRANSCRIPTS_DIR", str(tmp_path))
    monkeypatch.setenv("TRANSCRIPTS_ENABLED", "true")
    monkeypatch.setenv("AGENTS_RETENTION_DAYS", "30")
    T._store = None  # reset singleton so it picks up the tmp dir
    old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    T.get_store().record_turn(_rec(id="old", created_at=old))
    assert T.purge_older_than() == 1
    T._store = None
