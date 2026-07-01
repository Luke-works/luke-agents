"""The workflow agent returns the FULL WorkflowDoc each turn, repairs dangling
references so the graph is always wireable, and reports `changed` correctly."""
from fastapi.testclient import TestClient

import luke_agents.core.llm as llm
from luke_agents.agents.workflow_agent.agent import WorkflowAgent
from luke_agents.agents.workflow_agent.ops import repair_doc
from luke_agents.agents.workflow_agent.schema import WorkflowDocModel
from luke_agents.core.server import build_app


def _client(monkeypatch, doc: WorkflowDocModel) -> TestClient:
    monkeypatch.setattr(llm, "generate", lambda *a, **k: doc)
    monkeypatch.setattr(llm, "active_brain", lambda: "test")
    monkeypatch.setattr(llm, "active_model", lambda: "test-model")
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    return TestClient(build_app([WorkflowAgent()]))


def test_builds_a_workflow_and_reports_changed(monkeypatch):
    doc = WorkflowDocModel.model_validate({
        "id": "wf", "version": 1, "name": "Onboarding",
        "trigger": {"capability": "forms", "type": "form.submitted"},
        "nodes": [
            {"id": "n1", "kind": "action", "capability": "email", "action": "send", "next": "end"},
        ],
    })
    body = _client(monkeypatch, doc).post(
        "/agents/workflow/chat",
        json={"message": "email the customer when the form is submitted", "doc": None},
    ).json()
    assert body["changed"] is True
    assert body["doc"]["nodes"][0]["capability"] == "email"
    assert body["doc"]["nodes"][0]["next"] == "end"
    assert body["title"] == "Onboarding"


def test_unchanged_when_doc_matches(monkeypatch):
    current = {
        "id": "wf", "version": 1,
        "trigger": {"capability": "forms", "type": "form.submitted"},
        "nodes": [{"id": "n1", "kind": "action", "capability": "email", "action": "send", "next": "end"}],
    }
    # The model echoes the same doc back (a question / no-op edit).
    doc = WorkflowDocModel.model_validate(current)
    body = _client(monkeypatch, doc).post(
        "/agents/workflow/chat", json={"message": "what does this do?", "doc": current},
    ).json()
    assert body["changed"] is False


def test_repair_rewrites_dangling_references_to_end():
    doc = WorkflowDocModel.model_validate({
        "id": "wf", "version": 1,
        "trigger": {"capability": "forms", "type": "form.submitted"},
        "nodes": [
            {"id": "n1", "kind": "action", "capability": "email", "action": "send", "next": "n2"},
            {"id": "n2", "kind": "branch",
             "conditions": [{"expr": "amount > 10", "next": "ghost"}], "else": "n1"},
            {"id": "n3", "kind": "parallel", "branches": ["n1", "missing"], "join": "nope"},
        ],
    })
    fixed = repair_doc(doc)
    by_id = {n.id: n for n in fixed.nodes}
    assert by_id["n1"].next == "n2"                      # valid ref preserved
    assert by_id["n2"].conditions[0].next == "end"       # dangling -> end
    assert by_id["n2"].else_ == "n1"                     # valid ref preserved
    assert by_id["n3"].branches == ["n1", "end"]         # missing branch -> end
    assert by_id["n3"].join == "end"                     # dangling join -> end


def test_repair_dedupes_node_ids():
    doc = WorkflowDocModel.model_validate({
        "id": "wf", "version": 1,
        "trigger": {"capability": "forms", "type": "form.submitted"},
        "nodes": [
            {"id": "n1", "kind": "action", "capability": "email", "action": "send", "next": "end"},
            {"id": "n1", "kind": "action", "capability": "phone", "action": "call", "next": "end"},
        ],
    })
    fixed = repair_doc(doc)
    assert len(fixed.nodes) == 1
    assert fixed.nodes[0].capability == "email"  # first-seen wins
