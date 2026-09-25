"""Shared LLM brain — provider selection + a single typed entry point.

**The brain is chosen per REQUEST, not per process.** Each workspace brings its own
provider key (see ``credential.py``); core-engine decrypts it and attaches it to the turn,
and everything below runs on that credential — the workspace's provider, its model, its
bill. Only when no credential is attached do we fall back to the environment keys below,
which is the local-dev path (and is refused outright when AGENTS_REQUIRE_CREDENTIAL is on).

Interchangeable backends. For the env fallback the first one whose key is present wins
(Groq stays the default); set AGENTS_BRAIN to force a specific one:
  * Groq free/cheap tier -> used on Render / anywhere with GROQ_API_KEY (default).
  * OpenAI               -> GPT-5 nano (cheap, fast) when OPENAI_API_KEY is set.
  * Anthropic            -> Claude via ANTHROPIC_API_KEY (workspace-brought in practice).
  * Gemini free tier      -> used if GEMINI_API_KEY is set (blocked for managed domains).
  * Ollama                -> local open model in dev when no cloud key is set.

SECURITY: anything derived from a credential — provider SDK clients, circuit-breaker
state — MUST be keyed by the credential fingerprint, never by the bare brain name. Keyed
by brain alone, a cached client built with tenant A's key would serve tenant B's turn.

This module is agent-agnostic. An agent calls `generate(...)` with its own
system prompt, the user message, and the Pydantic model it wants back; we drive
whichever backend is active, force schema-shaped JSON, validate, and return the
typed object. All form/agent specifics live in the agent, not here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel


@dataclass(frozen=True)
class Usage:
    """LLM token usage for one generate() turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# Token usage from the most recent generate() in this request context, for the caller (e.g. the
# transcript recorder) to read without threading it through the typed return value. ContextVar keeps
# it async/request-safe.
_LAST_USAGE: ContextVar[Usage | None] = ContextVar("_last_usage", default=None)


def last_usage() -> Usage | None:
    """Token usage from the most recent generate() in this request context (None if unknown)."""
    return _LAST_USAGE.get()


def _note_usage(brain: str, model: str, prompt_tokens: object, completion_tokens: object) -> None:
    """Record LLM token usage for the turn: expose it via last_usage() AND count it in Prometheus.
    Best-effort — a provider may omit usage, and accounting must NEVER break a turn."""
    try:
        usage = Usage(prompt_tokens=int(prompt_tokens or 0), completion_tokens=int(completion_tokens or 0))
    except (TypeError, ValueError):
        usage = Usage()
    _LAST_USAGE.set(usage)
    try:
        from .metrics import TOKENS  # lazy import avoids any load-time import cycle

        if usage.prompt_tokens:
            TOKENS.labels(brain, model, "prompt").inc(usage.prompt_tokens)
        if usage.completion_tokens:
            TOKENS.labels(brain, model, "completion").inc(usage.completion_tokens)
    except Exception:  # noqa: BLE001
        pass  # metrics unavailable / label error — never fail the turn
    try:
        from .tokenbudget import record_current  # lazy: avoids any load-time import cycle

        # Charge this turn's tokens to the request's tenant (D5). Sums across multi-call turns;
        # no-op unless a per-tenant daily cap is armed. Best-effort — never fails the turn.
        record_current(usage.total_tokens)
    except Exception:  # noqa: BLE001
        pass

log = logging.getLogger("luke_agents.llm")

T = TypeVar("T", bound=BaseModel)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Best capability-per-dollar on Groq. Set GROQ_MODEL=moonshotai/kimi-k2-instruct
# for the premium option.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Falls back to this if the primary model errors (bad id, rate limit, bad JSON).
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GPT-5 nano: cheapest/fastest GPT-5 tier. Bump to gpt-5.4-nano for the newer
# snapshot. NOTE: nano is a reasoning model — it rejects `temperature`, so we
# never send it (see _openai). reasoning_effort is off by default (some nano
# variants 400 on it); set OPENAI_REASONING_EFFORT to opt in on models that allow it.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Claude Haiku: the cheap/fast tier, and strong at the structured-JSON work these agents do.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
# Anthropic requires an explicit output cap on every call; generous enough for a full form schema.
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Hard per-call timeout (seconds) applied to every provider client. Without it a
# hung upstream call blocks the worker indefinitely — a DoS on the single-worker
# Render setup. On timeout the SDK raises, the agent maps it to a 502, and the
# worker is freed.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

# Transient-retry + a lightweight per-brain circuit breaker (#24). A single blip (timeout, 5xx,
# dropped connection) on the active brain used to surface straight to the user as a 502; now the
# call is retried with backoff, and if a brain fails repeatedly the breaker opens to fail fast
# (and stop hammering a down provider) until a short cooldown elapses.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))                     # retries AFTER the first try
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "0.25"))  # exponential backoff base
LLM_BREAKER_THRESHOLD = int(os.getenv("LLM_BREAKER_THRESHOLD", "5"))         # consecutive fails → open
LLM_BREAKER_COOLDOWN_SECONDS = float(os.getenv("LLM_BREAKER_COOLDOWN_SECONDS", "15"))


class BrainUnavailable(RuntimeError):
    """The active brain's circuit is open (too many recent failures) — fail fast."""
    status_code = 503


# --------------------------------------------------------------------------- #
# Provider client reuse (#25) — construct each SDK client once and share it, so
# TCP/TLS connections and pools are reused across turns instead of rebuilt per call.
# --------------------------------------------------------------------------- #
# Bounded, because the cache key now includes the CREDENTIAL: one entry per distinct
# workspace key this worker has ever served, each owning an httpx client with its own
# connection pool, sockets and SSL context. Unbounded, a busy multi-tenant worker leaks
# file descriptors until it dies. LRU by insertion order; the evicted client is closed so
# its sockets go with it.
_CLIENT_CACHE_MAX = int(os.getenv("LLM_CLIENT_CACHE_MAX", "64"))

# A plain dict: insertion-ordered since 3.7, so it is all an LRU needs — and it keeps
# `_clients = {}` (what every test does to start clean) a valid reset.
_clients: dict = {}
_clients_lock = threading.Lock()


def _touch(key: str) -> None:
    """Move `key` to the most-recently-used end. Caller holds the lock."""
    try:
        _clients[key] = _clients.pop(key)
    except KeyError:  # evicted between the read and here — nothing to reorder
        pass


def _close_quietly(client) -> None:
    """Release an evicted client's sockets. Best-effort: a provider SDK that exposes no
    close() (or throws on it) must never break the turn that triggered the eviction."""
    for name in ("close", "_close"):
        fn = getattr(client, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.debug("llm: closing an evicted provider client failed", exc_info=True)
            return


def _cached_client(key: str, factory):
    with _clients_lock:
        client = _clients.get(key)
        if client is not None:
            _touch(key)  # mark recently used
            return client
    # Build outside the lock: constructing a client can do real work (TLS, DNS), and holding
    # the lock would serialise every tenant's first turn behind the slowest one.
    client = factory()
    evicted = None
    with _clients_lock:
        existing = _clients.get(key)
        if existing is not None:
            # Another thread won the race; keep theirs and discard ours.
            _touch(key)
            _close_quietly(client)
            return existing
        _clients[key] = client
        while len(_clients) > max(1, _CLIENT_CACHE_MAX):
            evicted = _clients.pop(next(iter(_clients)))
    if evicted is not None:
        _close_quietly(evicted)
    return client


# --------------------------------------------------------------------------- #
# Retry + circuit breaker (#24)
# --------------------------------------------------------------------------- #
# Keyed by SCOPE (brain + credential fingerprint), never by brain alone: one workspace's
# revoked or rate-limited key must not open the circuit for every other workspace on the
# same provider. `label` is the human-readable brain name, for messages only.
_breaker: dict = {}  # scope -> {"fails": int, "opened_at": float}
_breaker_lock = threading.Lock()


def _is_transient_llm(exc: BaseException) -> bool:
    """A provider-availability blip worth retrying — a timeout, a dropped connection, or a 5xx.
    A 429 is NOT retried here (the agent degrades it to a friendly 'busy, retry shortly'), and a
    validation/programming error is not transient."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return False
    if status in (408, 500, 502, 503, 504):
        return True
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(m in blob for m in
               ("timeout", "timed out", "connection", "connectionerror", "temporarily unavailable",
                "service unavailable", "reset by peer", "econnreset", "read error"))


def _breaker_gate(scope: str, label: str) -> None:
    with _breaker_lock:
        st = _breaker.get(scope)
        if st and st["fails"] >= LLM_BREAKER_THRESHOLD:
            if (time.monotonic() - st["opened_at"]) < LLM_BREAKER_COOLDOWN_SECONDS:
                raise BrainUnavailable(
                    f"{label} brain is temporarily unavailable (circuit open); retry shortly")
            st["fails"] = LLM_BREAKER_THRESHOLD - 1  # cooldown elapsed → half-open (allow one trial)


def _breaker_success(scope: str) -> None:
    with _breaker_lock:
        _breaker[scope] = {"fails": 0, "opened_at": 0.0}


def _breaker_failure(scope: str, label: str) -> None:
    with _breaker_lock:
        st = _breaker.setdefault(scope, {"fails": 0, "opened_at": 0.0})
        st["fails"] += 1
        if st["fails"] >= LLM_BREAKER_THRESHOLD:
            st["opened_at"] = time.monotonic()
            log.warning("llm: %s brain circuit OPEN after %d consecutive failures", label, st["fails"])


def _run_brain(brain: str, fn, scope: str | None = None):
    """Circuit-breaker gate + bounded transient-retry around one brain call (#24). Non-transient
    errors (validation, 429) surface immediately and do NOT trip the breaker; only exhausted
    transient failures count toward opening it.

    `scope` isolates the breaker to one credential; it defaults to the brain name so direct
    callers and tests keep the old process-wide behaviour."""
    scope = scope or brain
    _breaker_gate(scope, brain)
    attempts = 1 + max(0, LLM_MAX_RETRIES)
    last: BaseException | None = None
    for i in range(attempts):
        try:
            out = fn()
            _breaker_success(scope)
            return out
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_transient_llm(exc):
                raise  # not a provider-availability issue — don't retry or trip the breaker
            if i == attempts - 1:
                break
            log.warning("llm: transient %s on %s brain (attempt %d/%d): %s",
                        type(exc).__name__, brain, i + 1, attempts, exc)
            time.sleep(LLM_RETRY_BASE_SECONDS * (2 ** i))
    _breaker_failure(scope, brain)
    raise last  # type: ignore[misc]


def _credential():
    """This turn's workspace credential, or None for the env/platform fallback.
    Imported lazily so ``credential`` can import nothing from here (no cycle)."""
    from . import credential as _c

    return _c.current()


def _env_key(brain: str) -> str | None:
    """Platform key for a brain, from the environment — the local-dev fallback path.
    The dict is rebuilt per call so a monkeypatched/rotated module global is honoured."""
    return {
        "groq": GROQ_API_KEY,
        "openai": OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "gemini": GEMINI_API_KEY,
    }.get(brain)


def _default_model(brain: str) -> str:
    """The configured default model for a brain, used when the workspace picked none."""
    return {
        "groq": GROQ_MODEL,
        "openai": OPENAI_MODEL,
        "anthropic": ANTHROPIC_MODEL,
        "gemini": GEMINI_MODEL,
        "ollama": OLLAMA_MODEL,
    }.get(brain, "unknown")


def active_brain() -> str:
    """Which backend serves THIS turn.

    A workspace credential decides it outright. Otherwise (local dev) AGENTS_BRAIN forces
    one, else the first provider whose env key is present wins, Groq first."""
    cred = _credential()
    if cred is not None:
        return cred.provider
    override = os.getenv("AGENTS_BRAIN", "").strip().lower()
    if override:
        return override
    if GROQ_API_KEY:
        return "groq"
    if OPENAI_API_KEY:
        return "openai"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if GEMINI_API_KEY:
        return "gemini"
    return "ollama"


def active_model() -> str:
    """The model this turn runs on — the workspace's choice, else the brain's default.

    Best-effort for transcripts: if Groq fell back to GROQ_FALLBACK_MODEL on error, this
    still reports the primary — good enough for grouping training data by intent."""
    cred = _credential()
    if cred is not None and cred.model:
        return cred.model
    return _default_model(active_brain())


def generate(
    system: str,
    user: str,
    response_model: type[T],
    *,
    temperature: float = 0.3,
    model: str | None = None,
) -> T:
    """Run one turn against this request's brain and return a validated `response_model`.

    `system` is the agent's instructions, `user` the per-turn payload. Every backend is
    asked for schema-shaped JSON and the result is validated with Pydantic before
    returning, so callers always get a well-formed object (or an exception they can map to
    an HTTP error).

    `model` optionally overrides the brain's default for this one call — e.g. a cheap/fast
    model for high-volume classification. It applies ONLY on the env/platform path: it is
    a cost-tuning knob configured for one specific provider, and a workspace that brought
    its own account may be on a different one entirely (sending a Groq model id to
    Anthropic is a guaranteed 400). When a workspace credential is active, the workspace's
    own model choice wins — their provider, their model, their bill.
    """
    cred = _credential()
    brain = active_brain()
    if cred is None:
        # No workspace credential. Honour the BYO contract before touching a platform key.
        from .credential import CredentialRequired, require_credential

        if require_credential():
            raise CredentialRequired()
        api_key = _env_key(brain)
        chosen = model or _default_model(brain)
        allow_fallback = True
        scope = brain  # process-wide, as before: one platform key, one breaker
    else:
        api_key = cred.api_key
        chosen = cred.model or _default_model(brain)
        # Only substitute a different model when the workspace did not name one. If they
        # chose it, a silent switch bills them for a model they did not ask for; surfacing
        # the error lets the UI say "that model isn't available on your account".
        allow_fallback = cred.model is None
        # SECURITY: per-credential breaker scope. One workspace's revoked or throttled key
        # must never open the circuit for other workspaces on the same provider.
        scope = f"{brain}:{cred.fingerprint}"

    # #24: each brain call goes through the circuit-breaker + transient-retry wrapper.
    if brain == "groq":
        return _run_brain(brain, lambda: _groq(system, user, response_model, temperature, chosen,
                                               api_key=api_key, allow_fallback=allow_fallback), scope)
    if brain == "openai":  # nano ignores temperature
        return _run_brain(brain, lambda: _openai(system, user, response_model, chosen, api_key=api_key), scope)
    if brain == "anthropic":
        return _run_brain(brain, lambda: _anthropic(system, user, response_model, temperature, chosen,
                                                    api_key=api_key), scope)
    if brain == "gemini":
        return _run_brain(brain, lambda: _gemini(system, user, response_model, temperature, chosen,
                                                 api_key=api_key), scope)
    return _run_brain(brain, lambda: _ollama(system, user, response_model, temperature, chosen), scope)


def _client_key(brain: str, api_key: str | None) -> str:
    """Cache key for a provider SDK client.

    SECURITY: the key material is part of it. Caching on the brain name alone would let a
    client built with one workspace's key serve another workspace's turn — the exact
    cross-tenant leak BYO-key must not have. Hashed, never stored in clear."""
    digest = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:32]
    return f"{brain}:{digest}"


def _groq(system: str, user: str, response_model: type[T], temperature: float, model: str | None = None,
          *, api_key: str | None = None, allow_fallback: bool = True) -> T:
    from groq import Groq

    key = api_key if api_key is not None else GROQ_API_KEY
    client = _cached_client(_client_key("groq", key), lambda: Groq(api_key=key, timeout=LLM_TIMEOUT_SECONDS))  # #25
    # Groq's json_object response_format returns a 400 unless the word "json" appears somewhere
    # in the messages. Most prompts already describe a JSON output, but append a minimal
    # instruction for any that don't (e.g. the test-data prompt) so the request is never rejected.
    if "json" not in f"{system}\n{user}".lower():
        system = f"{system}\n\nRespond with a single JSON object."
    # The exact output shape lives in the agent's system prompt, so no verbose
    # JSON-schema dump here — keeps input tokens (and cost) down.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # Try the primary model, then fall back to a known-good one on any error
    # (unavailable model id, rate limit, malformed JSON, …).
    primary = model or GROQ_MODEL
    models = [primary]
    if allow_fallback and GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != primary:
        models.append(GROQ_FALLBACK_MODEL)

    last_err: Exception | None = None
    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            _u = getattr(resp, "usage", None)
            _note_usage("groq", model, getattr(_u, "prompt_tokens", None), getattr(_u, "completion_tokens", None))
            return response_model.model_validate_json(resp.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - try the next model
            last_err = exc
    raise last_err  # type: ignore[misc]


def _openai(system: str, user: str, response_model: type[T], model: str | None = None,
            *, api_key: str | None = None) -> T:
    """OpenAI GPT-5 nano via Structured Outputs (returns a validated Pydantic model).

    GPT-5 nano is a reasoning model: it rejects `temperature`, so we don't send it.
    `reasoning_effort` is sent only when OPENAI_REASONING_EFFORT is set, and we
    retry without it if the model rejects it (some nano variants 400 on it).
    """
    from openai import OpenAI

    key = api_key if api_key is not None else OPENAI_API_KEY
    client = _cached_client(_client_key("openai", key),
                            lambda: OpenAI(api_key=key, timeout=LLM_TIMEOUT_SECONDS))  # #25
    kwargs: dict = {
        "model": model or OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": response_model,  # Structured Outputs -> schema-shaped JSON
    }
    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT

    try:
        completion = client.beta.chat.completions.parse(**kwargs)
    except Exception:  # noqa: BLE001
        if "reasoning_effort" not in kwargs:
            raise
        kwargs.pop("reasoning_effort")  # model doesn't accept it — retry plainly
        completion = client.beta.chat.completions.parse(**kwargs)

    msg = completion.choices[0].message
    _u = getattr(completion, "usage", None)
    _note_usage("openai", kwargs["model"], getattr(_u, "prompt_tokens", None), getattr(_u, "completion_tokens", None))
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        return parsed
    # Fallback (e.g. refusal/edge): validate the raw JSON content ourselves.
    return response_model.model_validate_json(msg.content or "{}")


# Models that answered a forced `tool_choice` with a 400. Learned at runtime rather than listed,
# because the set changes whenever Anthropic ships a model and a hardcoded list would be wrong in
# whichever direction we guessed. Process-local and unbounded-by-design: it can only ever hold as
# many entries as the workspace has distinct Anthropic models.
_ANTHROPIC_NO_FORCED_TOOL: set[str] = set()


def _rejects_forced_tool(exc: Exception) -> bool:
    """Whether this is the provider saying "not that tool_choice", and nothing else.

    Deliberately narrow. A broad match would turn every 400 — a malformed schema, a bad model
    name, an oversized request — into a silent retry with weaker guarantees, which is worse than
    failing: the turn would come back shaped differently and the agent would act on it.
    """
    if getattr(exc, "status_code", None) not in (400, None):
        return False
    text = str(exc).lower()
    return "tool_choice" in text and ("not supported" in text or "unsupported" in text)


def _json_object(text: str) -> dict | None:
    """The first JSON object in a model's prose, or None.

    Only needed on the "auto" path, where the model may answer in text rather than calling the
    tool. Whatever comes back is still validated against the schema by the caller, so this is a
    parsing convenience and not a second, weaker contract.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        out = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _anthropic(system: str, user: str, response_model: type[T], _temperature: float,
               model: str | None = None, *, api_key: str | None = None) -> T:
    """Claude via forced tool use — the provider's way of guaranteeing a schema-shaped result.

    Anthropic has no `response_format`, so we declare the Pydantic schema as a single tool
    and force it with `tool_choice`. The model must answer by "calling" it, and the call's
    input IS the validated object. Costs the schema in input tokens, exactly like OpenAI's
    and Gemini's structured modes.
    """
    from anthropic import Anthropic

    key = api_key if api_key is not None else ANTHROPIC_API_KEY
    client = _cached_client(_client_key("anthropic", key),
                            lambda: Anthropic(api_key=key, timeout=LLM_TIMEOUT_SECONDS))  # #25
    chosen = model or ANTHROPIC_MODEL
    tool = {
        "name": "respond",
        "description": "Return the answer in the required shape. You must call this tool.",
        "input_schema": response_model.model_json_schema(),
    }

    def call(forced: bool):
        # NOTE: no `temperature`. anthropic 1.x removed it from messages.create and the method
        # takes no **kwargs, so passing it is a TypeError on every turn. Forced tool use already
        # pins the output shape, which is all `temperature` was doing for us on the other brains.
        return client.messages.create(
            model=chosen,
            max_tokens=ANTHROPIC_MAX_TOKENS,  # Anthropic requires an explicit output cap
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "respond"} if forced else {"type": "auto"},
        )

    # Forcing the tool is the strong path: the model MUST answer by calling it, so the shape is
    # guaranteed. But not every Claude model accepts a forced choice — the ones with extended
    # thinking reject `tool_choice` of type "tool" or "any" with a 400 — and a workspace picking
    # such a model got a hard failure on every single turn.
    #
    # So: force it where it works, remember where it does not, and fall back to "auto" for those.
    # Remembered per model because the answer is a property of the model, and paying a rejected
    # round trip on every turn to rediscover it is the kind of cost nobody sees until the bill.
    forced = chosen not in _ANTHROPIC_NO_FORCED_TOOL
    try:
        resp = call(forced)
    except Exception as exc:  # noqa: BLE001 - re-raised below unless it is THIS refusal
        if not (forced and _rejects_forced_tool(exc)):
            raise
        log.info("anthropic: %s does not accept a forced tool choice; using auto", chosen)
        _ANTHROPIC_NO_FORCED_TOOL.add(chosen)
        resp = call(False)

    _u = getattr(resp, "usage", None)
    _note_usage("anthropic", chosen, getattr(_u, "input_tokens", None), getattr(_u, "output_tokens", None))
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            return response_model.model_validate(block.input)
    # With a forced choice this is a provider fault. With "auto" the model was free to answer in
    # prose instead, so try the text before giving up — the schema is still the contract, and
    # model_validate below refuses anything that does not meet it.
    text = "".join(getattr(b, "text", "") or "" for b in (getattr(resp, "content", []) or []))
    parsed = _json_object(text)
    if parsed is not None:
        return response_model.model_validate(parsed)
    raise ValueError("anthropic returned no tool_use block")


def _gemini(system: str, user: str, response_model: type[T], temperature: float, model: str | None = None,
            *, api_key: str | None = None) -> T:
    from google import genai
    from google.genai import types

    key = api_key if api_key is not None else GEMINI_API_KEY
    client = _cached_client(_client_key("gemini", key), lambda: genai.Client(  # #25
        api_key=key,
        http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_SECONDS * 1000)),  # ms
    ))
    resp = client.models.generate_content(
        model=model or GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_model,  # forces schema-shaped JSON
            temperature=temperature,
        ),
    )
    _u = getattr(resp, "usage_metadata", None)
    _note_usage("gemini", model or GEMINI_MODEL,
                getattr(_u, "prompt_token_count", None), getattr(_u, "candidates_token_count", None))
    return response_model.model_validate_json(resp.text)


def _ollama(system: str, user: str, response_model: type[T], temperature: float, model: str | None = None) -> T:
    import ollama

    # No key: Ollama is a local daemon, so one cached client for the process is correct.
    resp = _cached_client("ollama", lambda: ollama.Client(timeout=LLM_TIMEOUT_SECONDS)).chat(  # #25
        model=model or OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=response_model.model_json_schema(),  # forces schema-shaped JSON
        options={"temperature": temperature},
    )
    _get = resp.get if hasattr(resp, "get") else (lambda k, r=resp: getattr(r, k, None))
    _note_usage("ollama", model or OLLAMA_MODEL, _get("prompt_eval_count"), _get("eval_count"))
    return response_model.model_validate_json(resp["message"]["content"])
