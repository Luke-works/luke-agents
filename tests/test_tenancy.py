"""Tenant isolation (#33): verified tenant resolution, per-tenant rate-limit
namespacing, tenant-stamped transcript writes, scoped export, and erasure."""
import json

import pytest
from fastapi import HTTPException

from luke_agents.agents.form_agent.agent import _rate_key
from luke_agents.core import ratelimit
from luke_agents.core.tenancy import default_tenant, resolve_tenant
from luke_agents.core.transcripts import JsonlStore, TurnRecord
from luke_agents.tools.export_finetune import _read_jsonl


class _Req:
    """Minimal fastapi.Request stand-in (only the headers/client tenancy reads)."""
    def __init__(self, headers=None, host="1.2.3.4"):
        self.headers = headers or {}

        class _C:
            def __init__(self, h): self.host = h
        self.client = _C(host)


# ── tenant resolution ───────────────────────────────────────────────────────
def test_resolve_uses_the_header():
    assert resolve_tenant(_Req(headers={"x-tenant-id": "acme"})) == "acme"


def test_resolve_falls_back_to_default_when_lenient(monkeypatch):
    monkeypatch.delenv("AGENTS_REQUIRE_TENANT", raising=False)
    assert resolve_tenant(_Req()) == default_tenant()


def test_resolve_treats_blank_and_null_as_absent():
    assert resolve_tenant(_Req(headers={"x-tenant-id": "  null  "})) == default_tenant()


def test_resolve_fails_closed_when_strict(monkeypatch):
    monkeypatch.setenv("AGENTS_REQUIRE_TENANT", "true")
    with pytest.raises(HTTPException) as ei:
        resolve_tenant(_Req())
    assert ei.value.status_code == 400


# ── rate-limit namespacing ──────────────────────────────────────────────────
def test_rate_key_is_namespaced_by_tenant():
    a = _rate_key(_Req(host="9.9.9.9"), "tenant-a")
    b = _rate_key(_Req(host="9.9.9.9"), "tenant-b")
    assert a == "form:t:tenant-a:ip:9.9.9.9"
    assert a != b  # same IP, different tenant => independent budget


def test_budgets_are_independent_across_tenants(monkeypatch):
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_MAX", 1)
    ratelimit.reset()  # fresh in-memory window for this test
    key_a = _rate_key(_Req(host="5.5.5.5"), "tenant-a")
    key_b = _rate_key(_Req(host="5.5.5.5"), "tenant-b")
    assert ratelimit.check_and_record(key_a)[0] is True
    assert ratelimit.check_and_record(key_a)[0] is False  # tenant-a now over budget
    assert ratelimit.check_and_record(key_b)[0] is True   # tenant-b unaffected


# ── transcript writes carry the tenant ──────────────────────────────────────
def _turn(turn_id, tenant):
    return TurnRecord(
        id=turn_id, agent="form", brain="test", model=None,
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        output={"reply": "ok"}, changed=True, tenant_id=tenant,
    )


def test_record_turn_persists_tenant_id(tmp_path):
    store = JsonlStore(str(tmp_path))
    store.init()
    store.record_turn(_turn("t1", "acme"))
    rows = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert rows[0]["tenant_id"] == "acme"


# ── export is tenant-scoped; cross-tenant rows are excluded ──────────────────
def test_export_reader_filters_by_tenant(tmp_path):
    store = JsonlStore(str(tmp_path))
    store.init()
    store.record_turn(_turn("a1", "acme"))
    store.record_turn(_turn("b1", "beta"))
    acme = list(_read_jsonl(str(tmp_path), agent=None, tenant="acme"))
    assert {r["id"] for r in acme} == {"a1"}  # beta excluded


# ── per-tenant erasure ──────────────────────────────────────────────────────
def test_delete_tenant_erases_only_that_tenant(tmp_path):
    store = JsonlStore(str(tmp_path))
    store.init()
    store.record_turn(_turn("a1", "acme"))
    store.record_turn(_turn("b1", "beta"))
    removed = store.delete_tenant("acme")
    assert removed == 1
    remaining = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines() if line.strip()]
    assert {r["id"] for r in remaining} == {"b1"}
