"""Guards the agents abuse-hardening fixes (PR #42):
- rate limit is keyed by IP, not the client-supplied user_id (bypass stays closed)
- request inputs are size-bounded
- feedback rating is bounded
"""
import pytest
from pydantic import ValidationError

from luke_agents.core import ratelimit
from luke_agents.agents.form_agent.agent import _rate_key
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
    assert _rate_key(_Req(host="9.9.9.9")) == "form:ip:9.9.9.9"


def test_rate_key_prefers_forwarded_for():
    r = _Req(headers={"x-forwarded-for": "203.0.113.5, 10.0.0.1"}, host="10.0.0.1")
    assert _rate_key(r) == "form:ip:203.0.113.5"


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
