"""An OUTBOUND form request steers the model with the two-party guidance (disabled display fields
vs required recipient fields); a normal request does not."""
from fastapi.testclient import TestClient

import luke_agents.core.llm as llm
from luke_agents.agents.form_agent.agent import FormAgent
from luke_agents.agents.form_agent.prompt import OUTBOUND_GUIDANCE
from luke_agents.agents.form_agent.schema import AssistantTurn
from luke_agents.core.server import build_app

SCHEMA = {"entities": {}, "root": []}


def _client(monkeypatch, captured: dict) -> TestClient:
    def fake_generate(system, user, model, **k):
        captured["system"] = system
        return AssistantTurn(reply="ok")

    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(llm, "active_brain", lambda: "test")
    monkeypatch.setattr(llm, "active_model", lambda: "test-model")
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    return TestClient(build_app([FormAgent()]))


def test_outbound_kind_appends_two_party_guidance(monkeypatch):
    captured: dict = {}
    client = _client(monkeypatch, captured)
    client.post("/chat", json={"message": "build a consent form", "schema": SCHEMA, "kind": "outbound"})
    assert OUTBOUND_GUIDANCE in captured["system"]
    assert "disabled" in captured["system"].lower()


def test_default_kind_omits_outbound_guidance(monkeypatch):
    captured: dict = {}
    client = _client(monkeypatch, captured)
    client.post("/chat", json={"message": "build a contact form", "schema": SCHEMA})
    assert OUTBOUND_GUIDANCE not in captured["system"]
