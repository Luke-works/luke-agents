"""D5 — per-tenant DAILY token cap. Proves: OFF by default (never blocks/counts dev/qa),
blocks a tenant once it hits today's ceiling, attributes usage to the enforced tenant (with
tenant isolation), the check-then-charge semantics, the Redis backend via fakeredis, and that
llm._note_usage auto-charges the budget so multi-call turns are summed."""
import fakeredis
import pytest
from fastapi.testclient import TestClient

import luke_agents.core.llm as llm
import luke_agents.core.tokenbudget as tb


@pytest.fixture(autouse=True)
def _clean_budget():
    """Each test starts with a fresh in-memory counter and no bound tenant."""
    tb.reset()
    yield
    tb.reset()


def test_disabled_by_default_never_blocks_or_counts():
    # No AGENTS_TENANT_DAILY_TOKEN_CAP env → disabled. enforce must not raise, record no-ops.
    tb.enforce("acme")  # binds tenant, no cap → returns
    tb.record_current(10_000_000)
    assert tb.current_usage("acme") == 0
    tb.enforce("acme")  # still fine after a huge (ignored) usage


def test_blocks_once_cap_reached(monkeypatch):
    monkeypatch.setenv("AGENTS_TENANT_DAILY_TOKEN_CAP", "1000")
    tb.enforce("acme")  # under cap (0) → allowed
    tb.record("acme", 1000)  # now at the ceiling
    with pytest.raises(Exception) as ei:  # HTTPException
        tb.enforce("acme")
    exc = ei.value
    assert getattr(exc, "status_code", None) == 429
    assert "daily AI usage limit" in exc.detail
    assert int(exc.headers["Retry-After"]) > 0


def test_check_then_charge_lets_the_crossing_turn_through(monkeypatch):
    monkeypatch.setenv("AGENTS_TENANT_DAILY_TOKEN_CAP", "1000")
    tb.enforce("acme")  # at 0 → allowed; the turn then runs and overshoots:
    tb.record("acme", 1500)  # 1500 > 1000
    assert tb.current_usage("acme") == 1500
    with pytest.raises(Exception):  # the NEXT turn is blocked
        tb.enforce("acme")


def test_record_current_attributes_to_enforced_tenant_and_is_isolated(monkeypatch):
    monkeypatch.setenv("AGENTS_TENANT_DAILY_TOKEN_CAP", "100000")
    tb.enforce("acme")  # binds "acme" to this context
    tb.record_current(300)
    tb.record_current(120)
    assert tb.current_usage("acme") == 420
    assert tb.current_usage("globex") == 0  # a different tenant is untouched


def test_redis_counter_increments_and_expires():
    client = fakeredis.FakeStrictRedis()
    counter = tb.RedisTokenCounter(client=client)
    counter.add("acme", "20260101", 100)
    counter.add("acme", "20260101", 50)
    assert counter.get("acme", "20260101") == 150
    assert counter.get("acme", "20260102") == 0  # a different day is a separate bucket
    assert client.ttl("agents:tokbudget:acme:20260101") > 0  # self-cleans


def test_note_usage_charges_the_current_tenant(monkeypatch):
    # Proves the llm.py wiring: after each generate(), _note_usage records the turn's tokens
    # against the request's tenant, so multi-call turns accumulate.
    monkeypatch.setenv("AGENTS_TENANT_DAILY_TOKEN_CAP", "100000")
    tb.enforce("acme")
    llm._note_usage("groq", "m", 30, 12)
    llm._note_usage("groq", "m", 40, 8)  # a second call in the same turn
    assert tb.current_usage("acme") == 90  # 42 + 48, SUMMED


def test_endpoint_blocks_second_call_when_tenant_over_cap(monkeypatch):
    """End-to-end through the default form /chat: a fake brain that charges 2000 tokens/turn
    trips the 1000-token cap, so the tenant's second call is rejected 429."""
    monkeypatch.setenv("AGENTS_TENANT_DAILY_TOKEN_CAP", "1000")
    monkeypatch.setenv("TRANSCRIPTS_ENABLED", "false")
    tb.reset()

    from luke_agents.agents.form_agent.schema import AssistantTurn

    def fake_generate(system, user, model_cls, **_kw):
        tb.record_current(2000)  # simulate the LLM layer metering this turn
        return AssistantTurn()

    monkeypatch.setattr(llm, "generate", fake_generate)

    import main

    client = TestClient(main.app)
    body = {"message": "hi", "schema": {"entities": {}, "root": []}}
    r1 = client.post("/chat", json=body)
    assert r1.status_code == 200
    r2 = client.post("/chat", json=body)
    assert r2.status_code == 429
    assert "daily AI usage limit" in r2.json()["detail"]
