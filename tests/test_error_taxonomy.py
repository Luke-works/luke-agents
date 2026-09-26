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


def test_out_of_credit_is_not_reported_as_a_passing_rate_limit(monkeypatch, tmp_path):
    """An exhausted account arrives AS a 429 — it must not be dressed up as "try again".

    The /chat handlers used to branch on 429 themselves and raise their own busy message,
    which ran BEFORE brain_http_error's exhausted check. A workspace out of credit was told
    to wait a few seconds, forever, and the X-AI-Credential signal core-engine acts on was
    lost with it. Both halves are asserted here because both were broken by the same line.
    """
    exc = RuntimeError("429 insufficient_quota: you exceeded your current quota")
    exc.status_code = 429
    r = _chat(_client(monkeypatch, tmp_path, _raise(exc)))

    assert r.status_code == 402, "out of credit is not a retry-in-a-moment condition"
    assert r.headers.get("X-AI-Credential") == "exhausted"
    assert "wait a few seconds" not in r.text.lower()
    assert "quota" in r.text.lower() or "credit" in r.text.lower()


def test_a_real_rate_limit_says_whose_limit_it_is(monkeypatch, tmp_path):
    """Under bring-your-own-key the quota belongs to the WORKSPACE, not to us.

    "The AI service is busy" sends someone to look at our status page over a limit only they
    can see or raise, and hides the remedy a second connected provider would give them.
    """
    r = _chat(_client(monkeypatch, tmp_path, _raise(RuntimeError("rate limit reached"))))
    assert r.status_code == 429
    body = r.text.lower()
    assert "your ai provider" in body
    assert "your own provider" in body or "rather than ours" in body


def test_a_provider_400_is_not_reported_as_a_passing_outage(monkeypatch, tmp_path):
    """Waiting does not fix a rejected request, so it must not be described as temporary.

    A workspace on a Claude model that refuses a forced `tool_choice` saw a 400 on every turn,
    reported as "temporarily unavailable; please retry shortly" — a retry loop against a
    permanent condition, with nothing pointing at the model as the thing to change.
    """
    exc = RuntimeError('tool_choice: type "tool" and "any" are not supported for this model.')
    exc.status_code = 400
    r = _chat(_client(monkeypatch, tmp_path, _raise(exc)))

    assert r.status_code == 422, "a rejected request is not an outage"
    body = r.text.lower()
    assert "temporarily unavailable" not in body
    assert "retry shortly" not in body
    # Names the lever the person actually has.
    assert "model" in body
    # And still never echoes the provider's own wording.
    assert "tool_choice" not in body


def test_a_timeout_says_it_timed_out(caplog):
    """"Unavailable" and "timed out" send a person to different places. The first says check the
    provider's status page; the second says the model was too slow for what you asked, and the
    levers are a faster model or a smaller request.

    A timeout carries NO status code, so it used to fall through into the generic 502 bucket with
    connection failures and unknown errors — which is exactly what a workspace saw after three
    30-second attempts against a slow Gemini model, with nothing telling them so.
    """
    import httpx

    from luke_agents.core.errors import brain_http_error

    for exc in (httpx.ReadTimeout("timed out"), TimeoutError("deadline exceeded")):
        mapped = brain_http_error(exc)
        assert mapped.status_code == 504, f"{type(exc).__name__} should be distinguishable"
        assert "too long" in str(mapped.detail)
        assert "unavailable" not in str(mapped.detail).lower()

    # A connection failure genuinely IS unavailable — the distinction has to cut both ways.
    assert brain_http_error(httpx.ConnectError("boom")).status_code == 502


def test_one_call_cannot_outlive_its_budget_by_retrying():
    """Retries MULTIPLY the timeout, and the arithmetic that justified the browser's wait forgot
    them: 3 attempts x 30s is 90s for one call, so a research turn (build + search + rebuild)
    could run ~206s against a client that gives up at 100. The turn could never report anything —
    the same failure the 25s client abort caused, rediscovered a scale up."""
    import luke_agents.core.llm as _llm

    budget = _llm.LLM_CALL_BUDGET_SECONDS
    worst_turn = 2 * budget + _llm.RESEARCH_TIMEOUT_SECONDS
    assert worst_turn < 100, (
        f"a research turn can take {worst_turn}s, past the client's 100s per-attempt budget "
        "(luke-consumer-ui agentTransport DEFAULT_TIMEOUTS)"
    )
    # And the budget must actually bound the retries, not just be declared.
    assert budget < (1 + _llm.LLM_MAX_RETRIES) * _llm.LLM_TIMEOUT_SECONDS
