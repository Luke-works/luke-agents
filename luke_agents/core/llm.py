"""Shared LLM brain — provider selection + a single typed entry point.

Interchangeable backends. By default the first one whose key is present wins
(Groq stays the prod default); set AGENTS_BRAIN to force a specific one:
  * Groq free/cheap tier -> used on Render / anywhere with GROQ_API_KEY (default).
  * OpenAI               -> GPT-5 nano (cheap, fast) when OPENAI_API_KEY is set.
  * Gemini free tier      -> used if GEMINI_API_KEY is set (blocked for managed domains).
  * Ollama                -> local open model in dev when no cloud key is set.

This module is agent-agnostic. An agent calls `generate(...)` with its own
system prompt, the user message, and the Pydantic model it wants back; we drive
whichever backend is active, force schema-shaped JSON, validate, and return the
typed object. All form/agent specifics live in the agent, not here.
"""
from __future__ import annotations

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
_clients: dict = {}
_clients_lock = threading.Lock()


def _cached_client(key: str, factory):
    client = _clients.get(key)
    if client is None:
        with _clients_lock:
            client = _clients.get(key)
            if client is None:
                client = factory()
                _clients[key] = client
    return client


# --------------------------------------------------------------------------- #
# Retry + circuit breaker (#24)
# --------------------------------------------------------------------------- #
_breaker: dict = {}  # brain -> {"fails": int, "opened_at": float}
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


def _breaker_gate(brain: str) -> None:
    with _breaker_lock:
        st = _breaker.get(brain)
        if st and st["fails"] >= LLM_BREAKER_THRESHOLD:
            if (time.monotonic() - st["opened_at"]) < LLM_BREAKER_COOLDOWN_SECONDS:
                raise BrainUnavailable(
                    f"{brain} brain is temporarily unavailable (circuit open); retry shortly")
            st["fails"] = LLM_BREAKER_THRESHOLD - 1  # cooldown elapsed → half-open (allow one trial)


def _breaker_success(brain: str) -> None:
    with _breaker_lock:
        _breaker[brain] = {"fails": 0, "opened_at": 0.0}


def _breaker_failure(brain: str) -> None:
    with _breaker_lock:
        st = _breaker.setdefault(brain, {"fails": 0, "opened_at": 0.0})
        st["fails"] += 1
        if st["fails"] >= LLM_BREAKER_THRESHOLD:
            st["opened_at"] = time.monotonic()
            log.warning("llm: %s brain circuit OPEN after %d consecutive failures", brain, st["fails"])


def _run_brain(brain: str, fn):
    """Circuit-breaker gate + bounded transient-retry around one brain call (#24). Non-transient
    errors (validation, 429) surface immediately and do NOT trip the breaker; only exhausted
    transient failures count toward opening it."""
    _breaker_gate(brain)
    attempts = 1 + max(0, LLM_MAX_RETRIES)
    last: BaseException | None = None
    for i in range(attempts):
        try:
            out = fn()
            _breaker_success(brain)
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
    _breaker_failure(brain)
    raise last  # type: ignore[misc]


def active_brain() -> str:
    """Which backend is live. AGENTS_BRAIN forces one (e.g. 'openai'); otherwise
    the first provider whose key is present wins (Groq stays the default)."""
    override = os.getenv("AGENTS_BRAIN", "").strip().lower()
    if override:
        return override
    if GROQ_API_KEY:
        return "groq"
    if OPENAI_API_KEY:
        return "openai"
    if GEMINI_API_KEY:
        return "gemini"
    return "ollama"


def active_model() -> str:
    """The configured primary model for the active brain. Best-effort for
    transcripts: if Groq fell back to GROQ_FALLBACK_MODEL on error, this still
    reports the primary — good enough for grouping training data by intent."""
    return {
        "groq": GROQ_MODEL,
        "openai": OPENAI_MODEL,
        "gemini": GEMINI_MODEL,
        "ollama": OLLAMA_MODEL,
    }.get(active_brain(), "unknown")


def generate(
    system: str,
    user: str,
    response_model: type[T],
    *,
    temperature: float = 0.3,
    model: str | None = None,
) -> T:
    """Run one turn against the active brain and return a validated `response_model`.

    `system` is the agent's instructions, `user` the per-turn payload. Every
    backend is asked for schema-shaped JSON and the result is validated with
    Pydantic before returning, so callers always get a well-formed object (or an
    exception they can map to an HTTP error).

    `model` optionally overrides the active brain's default model for this one
    call — e.g. a cheap/fast model for high-volume classification — without
    changing the global config for other agents. Leave it None to use the brain's
    configured default. Only meaningful for the brain that's actually active.
    """
    brain = active_brain()
    # #24: each brain call goes through the circuit-breaker + transient-retry wrapper.
    if brain == "groq":
        return _run_brain(brain, lambda: _groq(system, user, response_model, temperature, model))
    if brain == "openai":
        return _run_brain(brain, lambda: _openai(system, user, response_model, model))  # nano ignores temperature
    if brain == "gemini":
        return _run_brain(brain, lambda: _gemini(system, user, response_model, temperature, model))
    return _run_brain(brain, lambda: _ollama(system, user, response_model, temperature, model))


def _groq(system: str, user: str, response_model: type[T], temperature: float, model: str | None = None) -> T:
    from groq import Groq

    client = _cached_client("groq", lambda: Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS))  # #25
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
    if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != primary:
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


def _openai(system: str, user: str, response_model: type[T], model: str | None = None) -> T:
    """OpenAI GPT-5 nano via Structured Outputs (returns a validated Pydantic model).

    GPT-5 nano is a reasoning model: it rejects `temperature`, so we don't send it.
    `reasoning_effort` is sent only when OPENAI_REASONING_EFFORT is set, and we
    retry without it if the model rejects it (some nano variants 400 on it).
    """
    from openai import OpenAI

    client = _cached_client("openai", lambda: OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS))  # #25
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


def _gemini(system: str, user: str, response_model: type[T], temperature: float, model: str | None = None) -> T:
    from google import genai
    from google.genai import types

    client = _cached_client("gemini", lambda: genai.Client(  # #25
        api_key=GEMINI_API_KEY,
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
