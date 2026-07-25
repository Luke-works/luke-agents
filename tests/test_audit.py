"""#40 — append-only audit trail + operator RBAC on export.

Covers: the audit store round-trip, the /feedback handler emitting an audit entry with a
VERIFIED principal (the tenant, not a client-supplied field), the export tool writing a durable
audit row, and the operator credential gate on export.
"""
import pytest
from fastapi.testclient import TestClient

from luke_agents.agents.form_agent.agent import FormAgent
from luke_agents.core import transcripts as T
from luke_agents.core.server import build_app
from luke_agents.core.transcripts import AuditRecord, JsonlStore, list_audit, safe_record_audit
from luke_agents.tools import export_finetune as E


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    monkeypatch.setenv("AGENTS_TRANSCRIPT_RETRY_BASE_SECONDS", "0")
    yield


# --- audit store round-trip ---------------------------------------------------

def test_jsonl_audit_roundtrip_and_filter(tmp_path, monkeypatch):
    store = JsonlStore(str(tmp_path))
    store.init()
    monkeypatch.setattr(T, "_store", store)  # get_store() returns this

    safe_record_audit(AuditRecord(id="a1", actor="tenant:acme", action="feedback.label",
                                  target="t1", scope="acme"))
    safe_record_audit(AuditRecord(id="a2", actor="ops:bob", action="finetune.export",
                                  target="out.jsonl", scope="acme"))

    all_rows = list_audit()
    assert {r["action"] for r in all_rows} == {"feedback.label", "finetune.export"}
    only = list_audit(action="finetune.export")
    assert len(only) == 1 and only[0]["actor"] == "ops:bob"


def test_safe_record_audit_never_raises(monkeypatch):
    class Boom:
        def record_audit(self, rec):
            raise RuntimeError("audit store down")

    monkeypatch.setattr(T, "get_store", lambda: Boom())
    safe_record_audit(AuditRecord(id="x", actor="a", action="finetune.export"))  # must not raise


# --- /feedback emits an audit entry with a verified principal -----------------

def test_feedback_endpoint_emits_audit_with_tenant_principal(monkeypatch, tmp_path):
    store = JsonlStore(str(tmp_path))
    store.init()
    monkeypatch.setattr(T, "_store", store)
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    client = TestClient(build_app([FormAgent()]))

    # A client-supplied user_id must NOT become the actor.
    r = client.post("/feedback", json={"turn_id": "t-42", "accepted": True, "rating": 1,
                                       "user_id": "attacker-supplied"})
    assert r.status_code == 200

    rows = store.list_audit(action="feedback.label")
    assert len(rows) == 1
    a = rows[0]
    assert a["target"] == "t-42"
    assert a["actor"] == "tenant:public"      # verified principal (default tenant), not the user_id
    assert a["scope"] == "public"
    assert a["details"]["accepted"] is True and a["details"]["rating"] == 1


# --- export writes a durable audit row ----------------------------------------

def test_export_audit_writes_durable_row(monkeypatch):
    captured = []

    class Cap:
        def record_audit(self, rec):
            captured.append(rec)

    monkeypatch.setattr(T, "get_store", lambda: Cap())
    monkeypatch.setenv("AGENTS_ACTOR", "ops-alice")
    E._audit_export(action="export", tenant="acme", out="corpus.jsonl", kept=3, total=5)

    assert len(captured) == 1
    rec = captured[0]
    assert rec.action == "finetune.export" and rec.actor == "ops-alice"
    assert rec.scope == "acme" and rec.request_id  # a run id is attached


# --- operator RBAC gate on export ---------------------------------------------

def test_operator_gate_open_when_token_unset(monkeypatch):
    monkeypatch.delenv("AGENTS_OPERATOR_TOKEN", raising=False)
    assert E._require_operator(None)  # dev: runs (with a warning), returns an actor


def test_operator_gate_refuses_on_missing_or_wrong_token(monkeypatch):
    monkeypatch.setenv("AGENTS_OPERATOR_TOKEN", "s3cret")
    with pytest.raises(SystemExit):
        E._require_operator(None)
    with pytest.raises(SystemExit):
        E._require_operator("wrong")


def test_operator_gate_accepts_matching_token(monkeypatch):
    monkeypatch.setenv("AGENTS_OPERATOR_TOKEN", "s3cret")
    monkeypatch.setenv("AGENTS_ACTOR", "ops-carol")
    assert E._require_operator("s3cret") == "ops-carol"
