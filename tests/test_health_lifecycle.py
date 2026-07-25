"""#36 / #35 — lifespan startup/shutdown + liveness-vs-readiness health.

The app uses a lifespan context manager (not the deprecated on_event hooks); /health is a
can't-fail liveness signal and /health/ready actually checks dependencies (503 when a dep is down).
"""
from fastapi.testclient import TestClient

from luke_agents.agents.form_agent.agent import FormAgent
from luke_agents.core import transcripts as T
from luke_agents.core.server import build_app
from luke_agents.core.transcripts import JsonlStore


def _app():
    return build_app([FormAgent()])


def test_liveness_always_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "_store", JsonlStore(str(tmp_path)))
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    with TestClient(_app()) as client:  # `with` runs the lifespan (startup + shutdown)
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"


def test_readiness_ok_when_deps_healthy(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "_store", JsonlStore(str(tmp_path)))
    with TestClient(_app()) as client:
        r = client.get("/health/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["checks"] == {"transcripts": True, "brain": True}


def test_readiness_503_when_store_unreachable(monkeypatch, tmp_path):
    store = JsonlStore(str(tmp_path))
    monkeypatch.setattr(store, "ping", lambda: False)  # simulate the store being unreachable
    monkeypatch.setattr(T, "_store", store)
    with TestClient(_app()) as client:
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["ready"] is False


def test_lifespan_runs_startup_init(monkeypatch, tmp_path):
    calls = {"init": 0}

    class SpyStore(JsonlStore):
        def init(self):
            calls["init"] += 1

    monkeypatch.setattr(T, "_store", SpyStore(str(tmp_path)))
    with TestClient(_app()):
        pass  # entering the context runs lifespan startup; exiting runs shutdown
    assert calls["init"] == 1  # store initialized via the lifespan, not a deprecated on_event
