"""#37 — /v1 versioned API + a curated OpenAPI document, with legacy paths kept for compatibility."""
from fastapi.testclient import TestClient

import luke_agents.core.llm as llm
from luke_agents.agents.form_agent.agent import FormAgent
from luke_agents.agents.form_agent.schema import AssistantTurn
from luke_agents.core import transcripts as T
from luke_agents.core.server import build_app
from luke_agents.core.transcripts import JsonlStore

SCHEMA = {"entities": {}, "root": []}


def _client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setattr(llm, "generate", lambda *a, **k: AssistantTurn(reply="hi"))
    monkeypatch.setattr(llm, "active_brain", lambda: "test")
    monkeypatch.setattr(llm, "active_model", lambda: "test-model")
    monkeypatch.setattr(T, "_store", JsonlStore(str(tmp_path)))
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    return TestClient(build_app([FormAgent()], default_slug="form"))


def _chat(client, path):
    return client.post(path, json={"message": "hi", "schema": SCHEMA})


def test_versioned_v1_path_works(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    r = _chat(client, "/v1/agents/form/chat")
    assert r.status_code == 200 and r.json()["reply"] == "hi"


def test_legacy_unversioned_and_root_still_work(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    assert _chat(client, "/agents/form/chat").status_code == 200  # legacy unversioned
    assert _chat(client, "/chat").status_code == 200              # default agent at root


def test_openapi_is_curated(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    spec = client.get("/openapi.json").json()
    assert spec["info"]["version"] == "1.0.0"                     # not the stale 0.1.0
    scheme = spec["components"]["securitySchemes"]["AgentsApiKey"]
    assert scheme["type"] == "apiKey" and scheme["name"] == "X-Agents-Key"
    assert spec.get("servers")
    paths = spec["paths"]
    assert any(p.startswith("/v1/agents/form") for p in paths)    # versioned surface documented
    assert not any(p.startswith("/agents/form") for p in paths)   # legacy hidden from schema
    assert "/chat" in paths                                       # drop-in root documented
