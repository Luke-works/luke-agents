"""Guards the agents abuse-hardening fixes (PR #42):
- rate limit is keyed by IP, not the client-supplied user_id (bypass stays closed)
- request inputs are size-bounded
- feedback rating is bounded
"""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from luke_agents.core import ratelimit
from luke_agents.core.auth import require_api_key
from luke_agents.core.server import build_app
from luke_agents.agents.form_agent.agent import FormAgent, _rate_key
from luke_agents.agents.form_agent.schema import (
    ChatRequest,
    FeedbackRequest,
    TestDataRequest,
)


class _Client:
    def __init__(self, host):
        self.host = host


class _Req:
    """Minimal stand-in for fastapi.Request (only what _rate_key reads)."""
    def __init__(self, headers=None, host="1.2.3.4"):
        self.headers = headers or {}
        self.client = _Client(host)


def test_rate_key_is_ip_based():
    assert _rate_key(_Req(host="9.9.9.9"), "acme") == "form:t:acme:ip:9.9.9.9"


def test_rate_key_prefers_forwarded_for():
    r = _Req(headers={"x-forwarded-for": "203.0.113.5, 10.0.0.1"}, host="10.0.0.1")
    assert _rate_key(r, "acme") == "form:t:acme:ip:203.0.113.5"


def test_rate_limit_allows_then_blocks(monkeypatch):
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_MAX", 3)
    key = "test:bucket:allow-then-block"
    for _ in range(3):
        allowed, _retry = ratelimit.check_and_record(key)
        assert allowed is True
    allowed, retry = ratelimit.check_and_record(key)
    assert allowed is False
    assert retry >= 1


def test_chat_message_is_capped():
    ChatRequest(message="a normal message")  # ok
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 20_000)


def test_chat_schema_is_capped():
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", schema={"big": "y" * 300_000})


def test_feedback_rating_is_bounded():
    FeedbackRequest(turn_id="t", rating=1)  # ok
    with pytest.raises(ValidationError):
        FeedbackRequest(turn_id="t", rating=9)


def test_testdata_count_is_bounded():
    assert TestDataRequest().count == 1
    with pytest.raises(ValidationError):
        TestDataRequest(count=99)


def test_api_key_gate_is_noop_when_unset(monkeypatch):
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    require_api_key(_Req())  # must not raise


def test_api_key_gate_rejects_missing_or_wrong_key(monkeypatch):
    monkeypatch.setenv("AGENTS_API_KEY", "s3cret")
    with pytest.raises(HTTPException):
        require_api_key(_Req(headers={}))            # no key
    with pytest.raises(HTTPException):
        require_api_key(_Req(headers={"x-agents-key": "wrong"}))


def test_api_key_gate_accepts_correct_key(monkeypatch):
    monkeypatch.setenv("AGENTS_API_KEY", "s3cret")
    require_api_key(_Req(headers={"x-agents-key": "s3cret"}))  # must not raise


# --- #32: the gate is applied uniformly as a router-level dependency in build_app,
# so EVERY agent route (incl. any added later) is covered, while /health stays open.

def test_router_level_gate_blocks_all_agent_routes_when_key_set(monkeypatch):
    monkeypatch.setenv("AGENTS_API_KEY", "s3cret")
    client = TestClient(build_app([FormAgent()]))
    # Both the prefixed mount and the root mount are gated, no header → 401 before
    # the handler runs (so no LLM call / no rate-limit side effects).
    for path in ("/chat", "/agents/form/chat", "/testdata", "/feedback"):
        resp = client.post(path, json={"message": "hi"})
        assert resp.status_code == 401, f"{path} should require the API key"


def test_health_stays_open_even_with_key_set(monkeypatch):
    monkeypatch.setenv("AGENTS_API_KEY", "s3cret")
    client = TestClient(build_app([FormAgent()]))
    assert client.get("/health").status_code == 200


def test_routes_open_when_key_unset(monkeypatch):
    # Default-lenient: with no AGENTS_API_KEY, the gate is a no-op, so a missing
    # key does NOT 401 (the request proceeds past auth into normal handling).
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    client = TestClient(build_app([FormAgent()]))
    resp = client.post("/feedback", json={"turn_id": "nope"})
    assert resp.status_code != 401


# ── prod-profile fail-fast (enterprise hardening) ────────────────────────────
from luke_agents.core.server import assert_prod_hardened  # noqa: E402


def _clear_prod_env(monkeypatch):
    for k in ("AGENTS_ENV", "AGENTS_API_KEY", "AGENTS_CORS", "FORM_AGENT_CORS", "AGENTS_REQUIRE_TENANT"):
        monkeypatch.delenv(k, raising=False)


def test_prod_guard_noop_when_not_prod(monkeypatch):
    _clear_prod_env(monkeypatch)
    # No AGENTS_ENV → dev → never raises even though nothing is configured.
    assert_prod_hardened()
    monkeypatch.setenv("AGENTS_ENV", "dev")
    assert_prod_hardened()


def test_prod_guard_fails_open_service(monkeypatch):
    _clear_prod_env(monkeypatch)
    monkeypatch.setenv("AGENTS_ENV", "production")
    with pytest.raises(RuntimeError) as ei:
        assert_prod_hardened()
    msg = str(ei.value)
    assert "AGENTS_API_KEY" in msg and "AGENTS_CORS" in msg and "AGENTS_REQUIRE_TENANT" in msg


def test_prod_guard_passes_when_locked_down(monkeypatch):
    _clear_prod_env(monkeypatch)
    monkeypatch.setenv("AGENTS_ENV", "production")
    monkeypatch.setenv("AGENTS_API_KEY", "a-real-key")
    monkeypatch.setenv("AGENTS_CORS", "https://app.lukeflow.com")
    monkeypatch.setenv("AGENTS_REQUIRE_TENANT", "true")
    assert_prod_hardened()  # no raise
