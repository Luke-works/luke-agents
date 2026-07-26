"""#22 — Prometheus metrics at GET /metrics: request volume + latency + transcript counters."""
from fastapi.testclient import TestClient

from luke_agents.agents.form_agent.agent import FormAgent
from luke_agents.core import metrics as M
from luke_agents.core import transcripts as T
from luke_agents.core.server import build_app
from luke_agents.core.transcripts import JsonlStore


def _client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setattr(T, "_store", JsonlStore(str(tmp_path)))
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    return TestClient(build_app([FormAgent()]))


def test_metrics_endpoint_exposes_prometheus(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    client.get("/health")  # generate at least one request to count
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "agents_requests_total" in body
    assert "agents_request_latency_seconds" in body
    assert "agents_transcript_writes" in body  # durability counters surfaced


def test_request_counter_increments_per_request(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    before = M.REQUESTS.labels("GET", "/health", "200")._value.get()
    client.get("/health")
    after = M.REQUESTS.labels("GET", "/health", "200")._value.get()
    assert after == before + 1


def test_rate_limit_and_error_statuses_are_visible(monkeypatch, tmp_path):
    # A 404 (or any status) is recorded with its status label, so rate-limit 429s / LLM 502s show up.
    # Unmatched paths collapse to the "unmatched" route bucket (bounded cardinality — see below).
    client = _client(monkeypatch, tmp_path)
    before = M.REQUESTS.labels("GET", "unmatched", "404")._value.get()
    client.get("/nope")
    after = M.REQUESTS.labels("GET", "unmatched", "404")._value.get()
    assert after == before + 1


def test_unmatched_paths_do_not_explode_label_cardinality(monkeypatch, tmp_path):
    # The route label must be the matched TEMPLATE, never the raw path — otherwise an attacker
    # hitting /aaa, /aab, … mints one metric series per URL (a scrape/memory DoS).
    client = _client(monkeypatch, tmp_path)
    for p in ("/zzz-1", "/zzz-2", "/zzz-3", "/deep/random/path"):
        client.get(p)  # all 404
    body = client.get("/metrics").text
    assert 'route="unmatched"' in body
    for p in ("/zzz-1", "/zzz-2", "/zzz-3", "/deep/random/path"):
        assert f'route="{p}"' not in body  # no per-path series
