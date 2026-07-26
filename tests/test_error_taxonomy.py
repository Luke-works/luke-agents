"""A brain/LLM failure must map to a SAFE HTTP error: never echo the raw provider exception, and
preserve the status class (circuit-open 503 stays 503, rate limits become 429)."""
from fastapi.testclient import TestClient

import luke_agents.core.llm as llm
from luke_agents.agents.form_agent.agent import FormAgent
from luke_agents.core import transcripts as T
from luke_agents.core.server import build_app
from luke_agents.core.transcripts import JsonlStore

SCHEMA = {"entities": {}, "root": []}


def _client(monkeypatch, tmp_path, raiser) -> TestClient:
    monkeypatch.setattr(llm, "generate", raiser)
    monkeypatch.setattr(llm, "active_brain", lambda: "test")
    monkeypatch.setattr(llm, "active_model", lambda: "test-model")
    monkeypatch.setattr(T, "_store", JsonlStore(str(tmp_path)))
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    return TestClient(build_app([FormAgent()], default_slug="form"), raise_server_exceptions=False)


def _raise(exc):
    def _r(*_a, **_k):
        raise exc
    return _r


def _chat(client):
    return client.post("/chat", json={"message": "hi", "schema": SCHEMA})


def test_generic_brain_error_is_502_and_does_not_leak_the_exception(monkeypatch, tmp_path):
    secret = "groq org-xyz quota exceeded at https://api.groq.com/internal/secret"
    r = _chat(_client(monkeypatch, tmp_path, _raise(RuntimeError(secret))))
    assert r.status_code == 502
    # The raw provider message / internal URL must never reach the client.
    assert secret not in r.text
    assert "groq" not in r.text.lower()
    assert "api.groq.com" not in r.text


def test_circuit_open_preserves_503(monkeypatch, tmp_path):
    exc = RuntimeError("circuit open")
    exc.status_code = 503  # mimic the LLM circuit breaker's BrainUnavailable
    r = _chat(_client(monkeypatch, tmp_path, _raise(exc)))
    assert r.status_code == 503  # not flattened to 502


def test_rate_limit_maps_to_429(monkeypatch, tmp_path):
    r = _chat(_client(monkeypatch, tmp_path, _raise(RuntimeError("rate limit reached"))))
    assert r.status_code == 429
