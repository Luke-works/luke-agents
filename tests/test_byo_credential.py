"""BYO-key: each workspace runs the fleet on its OWN provider credential.

The risky part of per-request credentials is not the happy path, it's everything the
process used to share. Provider SDK clients and circuit-breaker state were cached under a
bare brain name ("groq"), which was correct while one platform key served every tenant and
becomes a cross-tenant leak the moment the key varies per request. These tests pin the
isolation properties, the refusal behaviour, and the fact that a key never escapes into a
log, a repr or an error.
"""
import asyncio

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

import luke_agents.core.credential as cred
from tests.aio import raises, record, returns, run, sequence

import luke_agents.core.llm as llm


class _R(BaseModel):
    ok: bool = True


class _Req:
    """Minimal stand-in for a Starlette Request — only .headers is read."""

    def __init__(self, **headers):
        self.headers = {k.replace("_", "-").lower(): v for k, v in headers.items()}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Credential binding and the client/breaker caches are process state — reset per test."""
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.setattr(llm, "_breaker", {})
    token = cred.set_current(None)
    yield
    cred.reset(token)


# --------------------------------------------------------------------------- #
# The key is secret material: it must not escape through repr/str/logs.
# --------------------------------------------------------------------------- #
def test_repr_and_str_never_contain_the_key():
    c = cred.Credential(provider="groq", api_key="gsk_super_secret_value", model="m")
    for rendered in (repr(c), str(c), f"{c}", "{}".format(c)):
        assert "gsk_super_secret_value" not in rendered
        assert "groq" in rendered  # still useful for debugging
    # An exception that interpolates the credential (a very easy mistake) stays clean too.
    assert "gsk_super_secret_value" not in str(RuntimeError(f"failed for {c}"))


def test_last_four_is_the_only_exposed_fragment():
    c = cred.Credential(provider="groq", api_key="abcdefgh1234")
    assert c.last_four == "1234"
    assert cred.Credential(provider="groq", api_key="ab").last_four == ""


def test_fingerprint_is_stable_distinct_and_not_reversible():
    a = cred.Credential(provider="groq", api_key="key-a")
    a2 = cred.Credential(provider="groq", api_key="key-a")
    b = cred.Credential(provider="groq", api_key="key-b")
    same_key_other_provider = cred.Credential(provider="openai", api_key="key-a")
    same_key_other_model = cred.Credential(provider="groq", api_key="key-a", model="other")

    assert a.fingerprint == a2.fingerprint          # stable
    assert a.fingerprint != b.fingerprint           # different key  -> different scope
    assert a.fingerprint != same_key_other_provider.fingerprint
    assert a.fingerprint != same_key_other_model.fingerprint
    assert "key-a" not in a.fingerprint             # not reversible


# --------------------------------------------------------------------------- #
# Resolving the credential off the request
# --------------------------------------------------------------------------- #
def test_from_request_reads_the_proxied_headers():
    c = cred.from_request(_Req(x_ai_provider="openai", x_ai_key="sk-abc", x_ai_model="gpt-5-nano"))
    assert (c.provider, c.api_key, c.model, c.source) == ("openai", "sk-abc", "gpt-5-nano", "tenant")


def test_from_request_returns_none_when_no_credential_is_attached():
    assert cred.from_request(_Req()) is None


def test_half_a_credential_is_a_400_not_a_silent_platform_fallback():
    """Falling back here would bill the wrong account for the turn."""
    for req in (_Req(x_ai_provider="groq"), _Req(x_ai_key="gsk-1")):
        with pytest.raises(HTTPException) as e:
            cred.from_request(req)
        assert e.value.status_code == 400


def test_unsupported_provider_is_rejected():
    with pytest.raises(HTTPException) as e:
        cred.from_request(_Req(x_ai_provider="evilcorp", x_ai_key="k"))
    assert e.value.status_code == 400
    # "ollama" is a local dev backend with no account behind it — not a BYO provider.
    with pytest.raises(HTTPException):
        cred.from_request(_Req(x_ai_provider="ollama", x_ai_key="k"))


def test_header_values_are_bounded_and_stripped():
    c = cred.from_request(_Req(x_ai_provider="  GROQ  ", x_ai_key="  gsk-1  ", x_ai_model="  m  "))
    assert (c.provider, c.api_key, c.model) == ("groq", "gsk-1", "m")
    huge = cred.from_request(_Req(x_ai_provider="groq", x_ai_key="k" * 5000))
    assert len(huge.api_key) == 512


def test_bind_refuses_a_credential_less_turn_only_when_required(monkeypatch):
    # This covers the REFUSAL decision only. asyncio.run() gives the coroutine a fresh
    # context, so what bind() binds does not survive back here — that the binding actually
    # reaches the LLM layer is proven end to end by
    # test_credential_headers_reach_the_llm_layer_on_every_mount, through a real request.
    monkeypatch.delenv("AGENTS_REQUIRE_CREDENTIAL", raising=False)
    asyncio.run(cred.bind(_Req()))         # lenient (local dev): no credential, no refusal

    monkeypatch.setenv("AGENTS_REQUIRE_CREDENTIAL", "true")
    with pytest.raises(cred.CredentialRequired) as e:
        asyncio.run(cred.bind(_Req()))
    # 402, not 403: the caller is allowed, the workspace just hasn't connected a provider.
    assert e.value.status_code == 402


# --------------------------------------------------------------------------- #
# The credential decides the brain and the model
# --------------------------------------------------------------------------- #
def test_active_brain_and_model_follow_the_credential(monkeypatch):
    monkeypatch.setattr(llm, "GROQ_API_KEY", "platform-key")  # env says groq...
    token = cred.set_current(cred.Credential(provider="anthropic", api_key="sk-ant", model="claude-x"))
    try:
        assert llm.active_brain() == "anthropic"              # ...credential wins
        assert llm.active_model() == "claude-x"
    finally:
        cred.reset(token)
    assert llm.active_brain() == "groq"                       # and is restored after


def test_credential_without_a_model_uses_the_provider_default(monkeypatch):
    monkeypatch.setattr(llm, "ANTHROPIC_MODEL", "claude-default")
    token = cred.set_current(cred.Credential(provider="anthropic", api_key="sk-ant"))
    try:
        assert llm.active_model() == "claude-default"
    finally:
        cred.reset(token)


# --------------------------------------------------------------------------- #
# THE leak test: nothing credential-derived may be shared across credentials.
# --------------------------------------------------------------------------- #
def _capture_groq(monkeypatch, seen):
    class FakeGroq:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key")
            seen.append(self.api_key)

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        async def create(self, **_kwargs):
            raise RuntimeError(f"called with {self.api_key}")

    monkeypatch.setattr("groq.AsyncGroq", FakeGroq)
    monkeypatch.setattr(llm, "GROQ_API_KEY", None)


def test_two_workspaces_never_share_a_provider_client(monkeypatch):
    """The core BYO-key guarantee. Cached under the bare brain name, workspace B's turn
    would be served by the client built with workspace A's key."""
    seen: list = []
    _capture_groq(monkeypatch, seen)

    for key in ("tenant-a-key", "tenant-b-key", "tenant-a-key"):
        token = cred.set_current(cred.Credential(provider="groq", api_key=key))
        try:
            with pytest.raises(RuntimeError):
                run(llm.generate("json", "u", _R))
        finally:
            cred.reset(token)

    # A and B each built their own client; A's second turn REUSED A's (so caching still works).
    assert seen == ["tenant-a-key", "tenant-b-key"]
    assert len(llm._clients) == 2
    assert all("tenant-a-key" not in k and "tenant-b-key" not in k for k in llm._clients), \
        "the cache key must hash the credential, never store it in clear"


def test_each_workspace_is_billed_on_its_own_key(monkeypatch):
    seen: list = []
    _capture_groq(monkeypatch, seen)
    token = cred.set_current(cred.Credential(provider="groq", api_key="tenant-b-key"))
    try:
        with pytest.raises(RuntimeError, match="tenant-b-key"):
            run(llm.generate("json", "u", _R))
    finally:
        cred.reset(token)


def test_one_workspaces_failures_do_not_open_another_workspaces_circuit(monkeypatch):
    """A revoked or throttled key is a per-workspace problem. Before per-credential
    scoping, five failures from one workspace took the provider down for everyone."""
    monkeypatch.setattr(llm, "LLM_BREAKER_THRESHOLD", 2)
    monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 0)

    async def boom():
        raise TimeoutError("provider timeout")  # transient -> counts toward the breaker

    a = cred.Credential(provider="groq", api_key="broken-key")
    b = cred.Credential(provider="groq", api_key="healthy-key")

    for _ in range(2):
        with pytest.raises(TimeoutError):
            run(llm._run_brain("groq", boom, f"groq:{a.fingerprint}"))

    # A is now open...
    with pytest.raises(llm.BrainUnavailable):
        run(llm._run_brain("groq", boom, f"groq:{a.fingerprint}"))
    # ...B is untouched and still gets to call the provider.
    with pytest.raises(TimeoutError):
        run(llm._run_brain("groq", boom, f"groq:{b.fingerprint}"))


def test_run_brain_still_defaults_to_a_process_wide_scope():
    """Back-compat: the env/platform path (and existing callers) keep one breaker per brain."""
    async def boom():
        raise TimeoutError("x")

    with pytest.raises(TimeoutError):
        run(llm._run_brain("groq", boom))
    assert "groq" in llm._breaker


# --------------------------------------------------------------------------- #
# A workspace's explicit model choice is honoured, not quietly substituted.
# --------------------------------------------------------------------------- #
def test_agent_model_override_is_ignored_when_the_workspace_chose_a_model(monkeypatch):
    """A per-call override (e.g. the sentiment agent's cheap model) is configured for ONE
    provider. Sending a Groq model id to Anthropic is a guaranteed 400, so the workspace's
    own choice wins."""
    captured: dict = {}

    async def fake_groq(system, user, response_model, temperature, model=None, *, api_key=None, allow_fallback=True):
        captured.update(model=model, allow_fallback=allow_fallback)
        return _R()

    monkeypatch.setattr(llm, "_groq", fake_groq)
    token = cred.set_current(cred.Credential(provider="groq", api_key="k", model="workspace-choice"))
    try:
        run(llm.generate("json", "u", _R, model="platform-cheap-model"))
    finally:
        cred.reset(token)
    assert captured["model"] == "workspace-choice"
    # ...and we must not silently swap in the platform's fallback model either: it would
    # bill them for a model they did not pick.
    assert captured["allow_fallback"] is False


def test_agent_model_override_still_applies_on_the_platform_path(monkeypatch):
    captured: dict = {}

    async def fake_groq(system, user, response_model, temperature, model=None, *, api_key=None, allow_fallback=True):
        captured.update(model=model, allow_fallback=allow_fallback)
        return _R()

    monkeypatch.setattr(llm, "_groq", fake_groq)
    monkeypatch.setattr(llm, "GROQ_API_KEY", "platform-key")
    monkeypatch.delenv("AGENTS_REQUIRE_CREDENTIAL", raising=False)
    run(llm.generate("json", "u", _R, model="platform-cheap-model"))
    assert captured["model"] == "platform-cheap-model"
    assert captured["allow_fallback"] is True


def test_groq_does_not_fall_back_to_another_model_when_the_workspace_picked_one(monkeypatch):
    tried: list = []

    class FakeGroq:
        def __init__(self, **_kwargs):
            pass

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        async def create(self, **kwargs):
            tried.append(kwargs["model"])
            raise RuntimeError("no such model")

    monkeypatch.setattr("groq.AsyncGroq", FakeGroq)
    monkeypatch.setattr(llm, "GROQ_FALLBACK_MODEL", "platform-fallback")

    with pytest.raises(RuntimeError):
        run(llm._groq("json", "u", _R, 0.3, "workspace-choice", api_key="k", allow_fallback=False))
    assert tried == ["workspace-choice"]

    tried.clear()
    with pytest.raises(RuntimeError):
        run(llm._groq("json", "u", _R, 0.3, "primary", api_key="k", allow_fallback=True))
    assert tried == ["primary", "platform-fallback"]


# --------------------------------------------------------------------------- #
# Refusal when no workspace credential is present
# --------------------------------------------------------------------------- #
def test_generate_refuses_rather_than_spending_a_platform_key(monkeypatch):
    monkeypatch.setenv("AGENTS_REQUIRE_CREDENTIAL", "true")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "platform-key")

    def explode(*_a, **_k):  # must never be reached
        raise AssertionError("the platform key was spent on a credential-less turn")

    monkeypatch.setattr(llm, "_groq", explode)
    with pytest.raises(cred.CredentialRequired):
        run(llm.generate("json", "u", _R))


# --------------------------------------------------------------------------- #
# The Anthropic brain (new): forced tool use gives a schema-shaped answer.
# --------------------------------------------------------------------------- #
class _Shaped(BaseModel):
    title: str
    count: int


def test_anthropic_forces_the_schema_and_returns_a_validated_object(monkeypatch):
    captured: dict = {}

    class _Block:
        type = "tool_use"
        input = {"title": "Contact form", "count": 3}

    class _Resp:
        content = [_Block()]
        usage = type("U", (), {"input_tokens": 11, "output_tokens": 7})()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["api_key"] = kwargs.get("api_key")

        @property
        def messages(self):
            return self

        async def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    monkeypatch.setattr("anthropic.AsyncAnthropic", FakeAnthropic)
    out = run(llm._anthropic("sys", "usr", _Shaped, 0.3, "claude-x", api_key="sk-ant-1"))

    assert out == _Shaped(title="Contact form", count=3)
    assert captured["api_key"] == "sk-ant-1"
    assert captured["model"] == "claude-x"
    assert captured["tool_choice"] == {"type": "tool", "name": "respond"}
    assert captured["tools"][0]["input_schema"] == _Shaped.model_json_schema()
    assert captured["max_tokens"] > 0          # Anthropic rejects a call without one
    assert captured["system"] == "sys"         # system prompt is a top-level arg, not a message
    assert llm.last_usage().total_tokens == 18


def test_anthropic_raises_rather_than_inventing_an_empty_object(monkeypatch):
    class _Resp:
        content = [type("B", (), {"type": "text", "text": "sorry"})()]
        usage = None

    class FakeAnthropic:
        def __init__(self, **_kwargs):
            pass

        @property
        def messages(self):
            return self

        async def create(self, **_kwargs):
            return _Resp()

    monkeypatch.setattr("anthropic.AsyncAnthropic", FakeAnthropic)
    with pytest.raises(ValueError, match="tool_use"):
        run(llm._anthropic("sys", "usr", _Shaped, 0.3, api_key="k"))


# --------------------------------------------------------------------------- #
# End to end: the credential core-engine attaches actually reaches the LLM layer,
# on every agent route, without any agent having to opt in.
# --------------------------------------------------------------------------- #
def _app(monkeypatch, tmp_path, seen):
    from fastapi.testclient import TestClient

    from luke_agents.agents.form_agent.agent import FormAgent
    from luke_agents.agents.form_agent.schema import AssistantTurn
    from luke_agents.core import transcripts as T
    from luke_agents.core.server import build_app
    from luke_agents.core.transcripts import JsonlStore

    async def _generate(*_a, **_k):
        seen.append(cred.current())      # what the agent's turn would have run on
        return AssistantTurn(reply="hi")

    monkeypatch.setattr(llm, "generate", _generate)
    monkeypatch.setattr(llm, "active_brain", lambda: "test")
    monkeypatch.setattr(llm, "active_model", lambda: "test-model")
    monkeypatch.setattr(T, "_store", JsonlStore(str(tmp_path)))
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    return TestClient(build_app([FormAgent()], default_slug="form"))


_BODY = {"message": "hi", "schema": {"entities": {}, "root": []}}


@pytest.mark.parametrize("path", ["/chat", "/agents/form/chat", "/v1/agents/form/chat"])
def test_credential_headers_reach_the_llm_layer_on_every_mount(monkeypatch, tmp_path, path):
    seen: list = []
    client = _app(monkeypatch, tmp_path, seen)
    res = client.post(path, json=_BODY, headers={
        "X-Tenant-Id": "acme",
        "X-AI-Provider": "anthropic",
        "X-AI-Key": "sk-ant-acme",
        "X-AI-Model": "claude-x",
    })
    assert res.status_code == 200, res.text
    assert len(seen) == 1
    assert (seen[0].provider, seen[0].api_key, seen[0].model) == ("anthropic", "sk-ant-acme", "claude-x")


def test_a_turn_without_a_credential_is_refused_when_byo_is_enforced(monkeypatch, tmp_path):
    seen: list = []
    monkeypatch.setenv("AGENTS_REQUIRE_CREDENTIAL", "true")
    client = _app(monkeypatch, tmp_path, seen)
    res = client.post("/chat", json=_BODY, headers={"X-Tenant-Id": "acme"})
    assert res.status_code == 402
    assert not seen, "the turn must be refused before any LLM call"


def test_the_credential_does_not_survive_into_the_next_request(monkeypatch, tmp_path):
    """ContextVars are per-request, but a regression here would silently bill one workspace
    for another's turn — the single worst outcome of this feature. Pin it."""
    seen: list = []
    client = _app(monkeypatch, tmp_path, seen)
    client.post("/chat", json=_BODY, headers={
        "X-Tenant-Id": "acme", "X-AI-Provider": "groq", "X-AI-Key": "gsk_acme"})
    client.post("/chat", json=_BODY, headers={"X-Tenant-Id": "other"})  # no credential
    assert seen[0].api_key == "gsk_acme"
    assert seen[1] is None, "workspace 'other' inherited acme's credential"


def test_health_reports_byo_mode_and_never_a_key(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTS_REQUIRE_CREDENTIAL", "true")
    client = _app(monkeypatch, tmp_path, [])
    body = client.get("/health").json()
    assert body["byo_key"] is True
    assert body["brain"] == "per-request"      # there is no process brain under BYO
    assert "key" not in str(body).lower().replace("byo_key", "")


# --------------------------------------------------------------------------- #
# Telling core-engine WHY a turn failed. It holds the key, so it is the only thing
# that can mark a workspace's provider invalid — but only for a real rejection.
# --------------------------------------------------------------------------- #
def _err(exc):
    from luke_agents.core.errors import brain_http_error

    return brain_http_error(exc)


class _Status(Exception):
    def __init__(self, status, msg=""):
        super().__init__(msg or f"status {status}")
        self.status_code = status


def test_a_rejected_key_is_signalled_as_invalid():
    for exc in (_Status(401), _Status(403),
                _Status(400, "Incorrect API key provided"),
                type("AuthenticationError", (Exception,), {})("nope")):
        e = _err(exc)
        assert e.status_code == 402, exc
        assert e.headers["X-AI-Credential"] == "invalid"
        assert "reject" in e.detail.lower()


def test_a_provider_outage_is_never_mistaken_for_a_bad_key():
    """The worst failure mode this feature has: disconnecting a working workspace because
    the provider had a bad minute."""
    for exc in (_Status(429, "rate limit exceeded"), _Status(500), _Status(503),
                TimeoutError("timed out"), _Status(408)):
        e = _err(exc)
        assert e.status_code != 402, exc
        assert "X-AI-Credential" not in (e.headers or {}), exc


def test_a_quota_message_on_a_401_is_still_a_rejection():
    """A 401 body often mentions quota. Mislabelled as 'busy', the workspace retries forever
    with no idea what is actually wrong."""
    e = _err(_Status(401, "You exceeded your current quota / rate limit"))
    assert e.status_code == 402
    assert e.headers["X-AI-Credential"] == "invalid"


def test_status_on_the_response_object_is_honoured():
    """google-genai puts the status on .response, not on the exception."""
    exc = RuntimeError("denied")
    exc.response = type("R", (), {"status_code": 403})()
    assert _err(exc).headers["X-AI-Credential"] == "invalid"


def test_missing_and_invalid_are_distinguishable(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTS_REQUIRE_CREDENTIAL", "true")
    client = _app(monkeypatch, tmp_path, [])
    res = client.post("/chat", json=_BODY, headers={"X-Tenant-Id": "acme"})
    assert res.status_code == 402
    assert res.headers["X-AI-Credential"] == "missing"   # never connected, not "key is bad"


def test_the_error_body_never_echoes_the_provider_message():
    """A provider's raw error can carry model ids, upstream URLs and account details."""
    e = _err(_Status(401, "key sk-live-abc123 is invalid for org org_secret"))
    assert "sk-live-abc123" not in e.detail
    assert "org_secret" not in e.detail


# --------------------------------------------------------------------------- #
# Regressions found by adversarial review. Each of these shipped green once.
# --------------------------------------------------------------------------- #
def test_anthropic_is_called_the_way_the_installed_sdk_actually_accepts():
    """anthropic 1.x removed `temperature` from messages.create and takes no **kwargs, so
    passing it is a TypeError on EVERY turn. The first fake accepted **kwargs, which hid it —
    so assert against the real installed signature instead of a stand-in."""
    import inspect

    from anthropic.resources.messages import Messages

    sig = inspect.signature(Messages.create)
    accepted = set(sig.parameters)
    takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())

    sent = {"model", "max_tokens", "system", "messages", "tools", "tool_choice"}
    unsupported = sent - accepted
    assert not unsupported or takes_kwargs, f"_anthropic sends {unsupported}, which the SDK rejects"
    assert "temperature" not in sent or "temperature" in accepted


def test_model_output_can_never_be_read_as_a_rejected_key(monkeypatch):
    """A pydantic ValidationError embeds the offending INPUT, which is model output. A form
    field labelled "Unauthorized access report" was enough to mark a perfectly good key
    INVALID and switch the workspace off."""
    from pydantic import ValidationError

    class _Strict(BaseModel):
        n: int

    try:
        _Strict.model_validate({"n": "unauthorized: invalid api key"})
        raise AssertionError("expected a validation error")
    except ValidationError as exc:
        e = _err(exc)
    assert e.status_code != 402
    assert "X-AI-Credential" not in (e.headers or {})


def test_a_bad_key_is_still_recognised_when_the_provider_uses_a_400():
    """Google answers a wrong key with 400 INVALID_ARGUMENT, not 401 — so status alone is not
    enough, and the text has to be consulted for client errors."""
    e = _err(_Status(400, "API key not valid. Please pass a valid API key."))
    assert e.status_code == 402
    assert e.headers["X-AI-Credential"] == "invalid"


def test_prose_on_a_server_error_is_never_a_rejected_key():
    """Text is only trusted on a client error; a 500 whose body happens to contain a phrase
    must not disconnect anyone."""
    e = _err(_Status(500, "upstream said: invalid api key (while proxying)"))
    assert e.status_code != 402


def test_an_exhausted_account_is_not_reported_as_a_passing_rate_limit():
    """Both usually arrive as 429, but one clears by waiting and the other never clears until
    someone pays. 'Wait a few seconds' forever is a mystery, not an error message."""
    for exc in (_Status(429, "You exceeded your current quota, please check your plan and billing details"),
                _Status(429, "insufficient_quota"),
                _Status(400, "Your credit balance is too low to access the Anthropic API")):
        e = _err(exc)
        assert e.status_code == 402, exc
        # NOT "invalid": the key works, so nothing should disconnect the workspace.
        assert e.headers["X-AI-Credential"] == "exhausted"
        assert "credit" in e.detail.lower() or "quota" in e.detail.lower()


def test_an_ordinary_rate_limit_is_still_a_429():
    e = _err(_Status(429, "rate limit exceeded, retry in 2s"))
    assert e.status_code == 429
    assert "X-AI-Credential" not in (e.headers or {})


def test_the_client_cache_is_bounded_and_closes_what_it_evicts(monkeypatch):
    """The cache key now includes the credential, so it grows with the number of distinct
    workspace keys this worker has served. Each entry owns a connection pool and sockets."""
    monkeypatch.setattr(llm, "_CLIENT_CACHE_MAX", 3)
    closed: list = []

    class _Client:
        def __init__(self, n):
            self.n = n

        def close(self):
            closed.append(self.n)

    for i in range(6):
        llm._cached_client(f"groq:key{i}", lambda i=i: _Client(i))

    assert len(llm._clients) == 3, "the cache must not grow with every workspace"
    assert closed == [0, 1, 2], "evicted clients must be closed, or their sockets leak"
    # The survivors are the three most recent.
    assert set(llm._clients) == {"groq:key3", "groq:key4", "groq:key5"}


def test_reuse_keeps_an_entry_alive_and_never_closes_a_live_client(monkeypatch):
    monkeypatch.setattr(llm, "_CLIENT_CACHE_MAX", 2)
    closed: list = []

    class _Client:
        def __init__(self, n):
            self.n = n

        def close(self):
            closed.append(self.n)

    a = llm._cached_client("groq:a", lambda: _Client("a"))
    llm._cached_client("groq:b", lambda: _Client("b"))
    assert llm._cached_client("groq:a", lambda: _Client("a2")) is a   # reused, not rebuilt
    llm._cached_client("groq:c", lambda: _Client("c"))                # evicts the LRU

    assert closed == ["b"], "recently used 'a' must survive; 'b' was the least recent"
    assert "groq:a" in llm._clients


# --------------------------------------------------------------------------- #
# The engine is the fleet's only caller, so per-user rate limiting depends on what
# it tells us — and on us believing it only when it has proved who it is.
# --------------------------------------------------------------------------- #
class _IpReq:
    def __init__(self, headers=None, peer="9.9.9.9"):
        self.headers = headers or {}

        class _C:
            host = peer

        self.client = _C()


def test_an_authenticated_engine_may_state_the_caller_ip(monkeypatch):
    from luke_agents.core.net import client_ip

    monkeypatch.setenv("AGENTS_API_KEY", "svc-secret")
    ip = client_ip(_IpReq({"x-agents-key": "svc-secret", "x-caller-ip": "203.0.113.7",
                           "x-forwarded-for": "10.0.0.1, 10.0.0.2"}))
    # Believed over the hop arithmetic: with the engine in front, the number of appended XFF
    # entries depends on the network path, and guessing wrong collapses every user in a
    # workspace into one rate-limit bucket.
    assert ip == "203.0.113.7"


def test_an_unauthenticated_caller_cannot_state_its_own_ip(monkeypatch):
    from luke_agents.core.net import client_ip

    monkeypatch.setenv("AGENTS_API_KEY", "svc-secret")
    for headers in ({"x-caller-ip": "1.2.3.4"},                                  # no service key
                    {"x-agents-key": "wrong", "x-caller-ip": "1.2.3.4"}):        # wrong key
        ip = client_ip(_IpReq({**headers, "x-forwarded-for": "8.8.8.8, 10.0.0.2"}))
        assert ip != "1.2.3.4", headers
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)
    assert client_ip(_IpReq({"x-caller-ip": "1.2.3.4", "x-forwarded-for": "8.8.8.8, 10.0.0.2"})) != "1.2.3.4"


def test_the_hop_counting_path_is_unchanged_when_no_caller_ip_is_stated(monkeypatch):
    from luke_agents.core.net import client_ip

    monkeypatch.setenv("AGENTS_API_KEY", "svc-secret")
    monkeypatch.setenv("AGENTS_TRUSTED_PROXY_HOPS", "2")
    assert client_ip(_IpReq({"x-agents-key": "svc-secret",
                             "x-forwarded-for": "spoofed, 203.0.113.9, 10.0.0.2"})) == "203.0.113.9"
