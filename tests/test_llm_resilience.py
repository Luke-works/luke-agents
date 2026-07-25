"""#24 / #25 — LLM transient-retry + circuit breaker, and provider-client reuse."""
import pytest

import luke_agents.core.llm as llm


class Transient(Exception):
    status_code = 503  # a retryable provider-availability error


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(llm, "_breaker", {})
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.setattr(llm, "LLM_RETRY_BASE_SECONDS", 0.0)  # no real sleeping in tests
    yield


# --- #25 client reuse ---------------------------------------------------------

def test_cached_client_constructs_once():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return object()

    a = llm._cached_client("groq", factory)
    b = llm._cached_client("groq", factory)
    assert a is b and calls["n"] == 1  # constructed once, reused


# --- #24 transient classification ---------------------------------------------

def test_transient_classification():
    assert llm._is_transient_llm(Transient())                        # 503
    assert llm._is_transient_llm(RuntimeError("request timed out"))  # timeout text

    class RateLimit(Exception):
        status_code = 429

    assert not llm._is_transient_llm(RateLimit())        # 429 surfaces (agent degrades it)
    assert not llm._is_transient_llm(ValueError("bad json"))  # validation is not transient


# --- #24 retry-with-backoff ---------------------------------------------------

def test_run_brain_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 3)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        if n["c"] < 3:
            raise Transient()
        return "ok"

    assert llm._run_brain("groq", fn) == "ok"
    assert n["c"] == 3


def test_run_brain_does_not_retry_or_trip_breaker_on_non_transient(monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 3)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise ValueError("validation error")

    with pytest.raises(ValueError):
        llm._run_brain("groq", fn)
    assert n["c"] == 1                                  # not retried
    assert llm._breaker.get("groq", {}).get("fails", 0) == 0  # breaker untouched


# --- #24 circuit breaker ------------------------------------------------------

def test_breaker_opens_and_fails_fast(monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 0)
    monkeypatch.setattr(llm, "LLM_BREAKER_THRESHOLD", 3)
    monkeypatch.setattr(llm, "LLM_BREAKER_COOLDOWN_SECONDS", 60)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise Transient()

    for _ in range(3):
        with pytest.raises(Transient):
            llm._run_brain("groq", fn)
    assert calls["n"] == 3
    # Circuit is now open → fail fast WITHOUT invoking fn again.
    with pytest.raises(llm.BrainUnavailable):
        llm._run_brain("groq", fn)
    assert calls["n"] == 3


def test_breaker_resets_on_success(monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 0)
    monkeypatch.setattr(llm, "LLM_BREAKER_THRESHOLD", 3)
    state = {"fail": True}

    def fn():
        if state["fail"]:
            raise Transient()
        return "ok"

    with pytest.raises(Transient):
        llm._run_brain("g", fn)          # fails = 1
    state["fail"] = False
    assert llm._run_brain("g", fn) == "ok"
    assert llm._breaker["g"]["fails"] == 0  # a success clears the failure count
