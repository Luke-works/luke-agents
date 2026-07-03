"""The form agent surfaces a LIFECYCLE action (check in / publish / undo checkout) for the app
to run, and when it does it leaves the form untouched — even if the model also emitted field ops."""
from fastapi.testclient import TestClient

import luke_agents.core.llm as llm
from luke_agents.agents.form_agent.agent import FormAgent
from luke_agents.agents.form_agent.schema import AssistantTurn, FormOp, SpecField
from luke_agents.core.server import build_app

SCHEMA = {"entities": {}, "root": []}


def _client(monkeypatch, turn: AssistantTurn) -> TestClient:
    monkeypatch.setattr(llm, "generate", lambda *a, **k: turn)
    monkeypatch.setattr(llm, "active_brain", lambda: "test")
    monkeypatch.setattr(llm, "active_model", lambda: "test-model")
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    return TestClient(build_app([FormAgent()]))


def test_publish_action_is_surfaced_and_form_untouched(monkeypatch):
    client = _client(monkeypatch, AssistantTurn(reply="Publishing this for you.", action="publish"))
    body = client.post("/chat", json={"message": "make it live", "schema": SCHEMA}).json()
    assert body["action"] == "publish"
    assert body["changed"] is False


def test_action_ignores_any_field_ops(monkeypatch):
    # Even if the model wrongly pairs an op with an action, the form must be left untouched.
    turn = AssistantTurn(
        operations=[FormOp(op="add", field=SpecField(key="x", label="X"))],
        reply="Checking in.", action="checkin",
    )
    body = _client(monkeypatch, turn).post("/chat", json={"message": "check in", "schema": SCHEMA}).json()
    assert body["action"] == "checkin"
    assert body["changed"] is False
    assert body["schema"] == SCHEMA  # no field added


def test_a_normal_edit_carries_no_action(monkeypatch):
    turn = AssistantTurn(
        operations=[FormOp(op="add", field=SpecField(key="email", label="Email", type="email"))],
        reply="Added an email field.",
    )
    body = _client(monkeypatch, turn).post("/chat", json={"message": "add an email field", "schema": SCHEMA}).json()
    assert body["action"] is None
    assert body["changed"] is True
