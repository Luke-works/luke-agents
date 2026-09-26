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

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from functools import lru_cache
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


# Token usage for THIS REQUEST, for the caller (e.g. the transcript recorder) to read without
# threading it through the typed return value. ContextVar keeps it async/request-safe.
#
# CUMULATIVE, not last-write-wins. A turn was one provider call when this was written; a research
# turn is three — the build that asks for a fact, the search, then the rebuild with the findings —
# and overwriting meant the transcript, which is the durable per-tenant usage record, captured
# only the last of them. The daily budget and the Prometheus counters were always charged for all
# three (`record_current` sums, `.inc()` accumulates); it was the auditable record that
# under-reported, which is the one a bill would be argued from.
_LAST_USAGE: ContextVar[Usage | None] = ContextVar("_last_usage", default=None)


def last_usage() -> Usage | None:
    """Token usage across every provider call made in this request context (None if unknown)."""
    return _LAST_USAGE.get()


def reset_usage() -> None:
    """Start a new request's accounting. Each request runs on its own ContextVar copy in a
    threadpool worker, but a worker THREAD is reused, so without this a turn could inherit a
    previous turn's total on the same thread."""
    _LAST_USAGE.set(None)


def _note_usage(brain: str, model: str, prompt_tokens: object, completion_tokens: object) -> None:
    """Record LLM token usage for the turn: expose it via last_usage() AND count it in Prometheus.
    Best-effort — a provider may omit usage, and accounting must NEVER break a turn."""
    try:
        call = Usage(prompt_tokens=int(prompt_tokens or 0), completion_tokens=int(completion_tokens or 0))
    except (TypeError, ValueError):
        call = Usage()

    # THIS call's tokens go to the meters, which accumulate on their own — a Prometheus counter
    # is `.inc()`d and the daily budget is `record_current`ed, both additive. The running TOTAL
    # goes only to the ContextVar the transcript reads. Sending the running total to the meters
    # instead charges call 1 again inside call 2's figure: a two-call turn of 42 + 48 billed 132
    # rather than 90, every research turn silently overcharging the workspace's daily cap.
    prior = _LAST_USAGE.get()
    _LAST_USAGE.set(Usage(
        prompt_tokens=(prior.prompt_tokens if prior else 0) + call.prompt_tokens,
        completion_tokens=(prior.completion_tokens if prior else 0) + call.completion_tokens,
    ))
    usage = call
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

# Web research (see `research`). A cap, not a budget: the providers bill per search — Anthropic
# publishes $10 per 1,000 — and an uncapped agent that decides to "check a few more sources" spends
# a workspace's money without anyone watching. Three is enough for "what is on this restaurant's
# menu" and small enough that a runaway turn is a rounding error.
#
# HONEST LIMIT: enforceable on Anthropic (`max_uses` on the tool) and OpenAI (`max_tool_calls` on
# the request). Gemini has NO equivalent — `types.GoogleSearch` exposes only search_types,
# blocking_confidence, exclude_domains and time_range_filter — so on Gemini a turn is bounded by
# RESEARCH_TIMEOUT_SECONDS and RESEARCH_MAX_TOKENS and not by a number of searches. Claiming
# otherwise in a comment would be worse than the gap.
RESEARCH_MAX_USES = int(os.getenv("RESEARCH_MAX_USES", "3"))
# Research reads whole web pages, so it needs more room than a form-schema turn and more time than
# a single completion: the provider runs several searches inside one call.
RESEARCH_MAX_TOKENS = int(os.getenv("RESEARCH_MAX_TOKENS", "4096"))
# 25s, NOT 90. A research turn is THREE provider calls inside ONE /chat request — the build that
# asks, the search, then the rebuild with findings — and the browser aborts a single request after
# 25s (ATTEMPT_TIMEOUT_MS in luke-consumer-ui's formAgentApi.ts). At 90 the worst case was 30 + 90
# + 30 = 150s: a research turn could NEVER finish before the client gave up, and the client then
# retried it, starting a fresh billable search nobody would ever see. Keep the worst case
# (LLM_TIMEOUT_SECONDS + this + LLM_TIMEOUT_SECONDS) comfortably under that client budget.
RESEARCH_TIMEOUT_SECONDS = float(os.getenv("RESEARCH_TIMEOUT_SECONDS", "25"))

# A ceiling on provider calls IN FLIGHT at once, per worker.
#
# The threadpool used to be this bound, implicitly: a sync endpoint held a slot for its whole
# provider call, so the pool size capped concurrency whether anyone meant it to or not. Awaiting
# instead removes that accidental limit entirely — a burst can open as many upstream connections
# as requests arrive, and the failure mode is the provider rate-limiting the workspace's own key,
# or the box running out of sockets. A semaphore puts the bound back, deliberately this time and
# at a number chosen rather than inherited.
LLM_MAX_INFLIGHT = int(os.getenv("LLM_MAX_INFLIGHT", "64"))
_inflight: "asyncio.Semaphore | None" = None
_inflight_loop = None


def _inflight_gate() -> "asyncio.Semaphore":
    """The per-loop semaphore. Built lazily because an asyncio primitive binds to the running
    loop, and this module is imported long before one exists (and, in tests, across several)."""
    global _inflight, _inflight_loop
    loop = asyncio.get_running_loop()
    if _inflight is None or _inflight_loop is not loop:
        _inflight = asyncio.Semaphore(max(1, LLM_MAX_INFLIGHT))
        _inflight_loop = loop
    return _inflight


# Transient-retry + a lightweight per-brain circuit breaker (#24). A single blip (timeout, 5xx,
# dropped connection) on the active brain used to surface straight to the user as a 502; now the
# call is retried with backoff, and if a brain fails repeatedly the breaker opens to fail fast
# (and stop hammering a down provider) until a short cooldown elapses.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))                     # retries AFTER the first try
# A ceiling on ONE generate() call INCLUDING its retries.
#
# Retries multiply the timeout, and the budget arithmetic that justified the client's wait forgot
# them: 3 attempts x 30s is 90s for a single call, so a research turn (build + search + rebuild)
# could run 206s against a browser that gives up at 100. A turn that times out therefore could
# never report anything — the same failure the 25s client abort caused, rediscovered at a bigger
# scale, because "worst case" was computed from the per-attempt timeout alone.
#
# 35s keeps a research turn at 35 + RESEARCH_TIMEOUT + 35 = 95s, inside the client's 100. Raise
# this and the client budget in luke-consumer-ui's agentTransport must move with it.
LLM_CALL_BUDGET_SECONDS = float(os.getenv("LLM_CALL_BUDGET_SECONDS", "35"))
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


#: Strong references to in-flight client-close tasks (see `_close_quietly`). asyncio holds only
#: a weak reference to a task, so anything fire-and-forget must be kept alive by its creator or it
#: may simply vanish before it runs.
_CLOSING: set = set()


def _close_quietly(client) -> None:
    """Release an evicted client's sockets. Best-effort: a provider SDK that exposes no
    close() (or throws on it) must never break the turn that triggered the eviction.

    The async SDK clients return a COROUTINE from close(). Calling that and dropping it would
    close nothing — the sockets would leak exactly as they did before this cache was bounded —
    and would raise "coroutine was never awaited" on the way out. Eviction happens inside a
    turn that is already on the event loop, so hand the coroutine to the loop and let it finish
    in the background; nobody is waiting on a socket teardown.
    """
    # EVERY transport the client owns, not just the first one found. google-genai keeps two:
    # `close()` shuts the sync transport, while the async one this code actually calls lives at
    # `.aio.aclose()` — closing only the first leaks precisely the sockets in use. Stopping at
    # the first match looked tidy and released the wrong half.
    closers = [getattr(client, n, None) for n in ("close", "aclose", "_close")]
    aio = getattr(client, "aio", None)
    if aio is not None:
        closers += [getattr(aio, n, None) for n in ("aclose", "close")]

    for fn in closers:
        if not callable(fn):
            continue
        try:
            out = fn()
            if asyncio.iscoroutine(out):
                try:
                    task = asyncio.get_running_loop().create_task(out)
                    # HOLD A REFERENCE. asyncio keeps only a weak one, so a bare create_task can
                    # be garbage-collected mid-flight and the close never happens — the sockets
                    # leak exactly as they did before this cache was bounded, and silently.
                    _CLOSING.add(task)
                    task.add_done_callback(_CLOSING.discard)
                except RuntimeError:
                    # No loop (a test, or shutdown): run it to completion here instead.
                    asyncio.run(out)
        except Exception:  # noqa: BLE001
            log.debug("llm: closing an evicted provider client failed", exc_info=True)


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


async def _run_brain(brain: str, fn, scope: str | None = None):
    """Circuit-breaker gate + bounded transient-retry around one brain call (#24). Non-transient
    errors (validation, 429) surface immediately and do NOT trip the breaker; only exhausted
    transient failures count toward opening it.

    `scope` isolates the breaker to one credential; it defaults to the brain name so direct
    callers and tests keep the old process-wide behaviour."""
    scope = scope or brain
    _breaker_gate(scope, brain)
    attempts = 1 + max(0, LLM_MAX_RETRIES)
    gate = _inflight_gate()
    started = time.monotonic()
    last: BaseException | None = None
    for i in range(attempts):
        try:
            async with gate:  # bound the provider calls in flight; see LLM_MAX_INFLIGHT
                out = await fn()
            _breaker_success(scope)
            return out
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_transient_llm(exc):
                raise  # not a provider-availability issue — don't retry or trip the breaker
            if i == attempts - 1:
                break
            # Stop retrying once another attempt cannot finish inside the budget. Without this
            # the retry count silently multiplies the timeout and the caller — a browser with a
            # deadline of its own — gives up before we ever answer.
            spent = time.monotonic() - started
            if spent + LLM_TIMEOUT_SECONDS > LLM_CALL_BUDGET_SECONDS:
                log.warning("llm: %s brain out of budget after %.1fs (%d attempt(s)); not retrying",
                            brain, spent, i + 1)
                break
            log.warning("llm: transient %s on %s brain (attempt %d/%d): %s",
                        type(exc).__name__, brain, i + 1, attempts, exc)
            # asyncio.sleep, not time.sleep: this runs ON the event loop now, and a
            # blocking sleep here would stall every other request in the worker.
            await asyncio.sleep(LLM_RETRY_BASE_SECONDS * (2 ** i))
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


async def generate(
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
        return await _run_brain(brain, lambda: _groq(system, user, response_model, temperature, chosen,
                                               api_key=api_key, allow_fallback=allow_fallback), scope)
    if brain == "openai":  # nano ignores temperature
        return await _run_brain(brain, lambda: _openai(system, user, response_model, chosen, api_key=api_key), scope)
    if brain == "anthropic":
        return await _run_brain(brain, lambda: _anthropic(system, user, response_model, temperature, chosen,
                                                    api_key=api_key), scope)
    if brain == "gemini":
        return await _run_brain(brain, lambda: _gemini(system, user, response_model, temperature, chosen,
                                                 api_key=api_key), scope)
    return await _run_brain(brain, lambda: _ollama(system, user, response_model, temperature, chosen), scope)


@dataclass(frozen=True)
class Research:
    """What a web-research pass found: prose for the model, sources for the person."""

    text: str
    #: ``{"url", "title"}`` per cited source, in the order the provider returned them.
    sources: list[dict]


#: Providers with a FIRST-PARTY search tool. Groq and Ollama have none, and giving them one would
#: mean holding a search vendor's key ourselves - which breaks the rule that the work runs on the
#: workspace's account, not ours. They simply build without research, exactly as before.
RESEARCH_BRAINS = frozenset({"anthropic", "openai", "gemini"})

RESEARCH_SYSTEM = (
    "You are researching a specific factual question so that a colleague can build a form from "
    "the answer. Search the web when the question depends on current or local facts you cannot "
    "know; answer directly when it does not.\n\n"
    "Report ONLY what the sources actually say. If you cannot find something - a menu, a price, "
    "opening hours - say so plainly instead of producing a plausible substitute; an invented menu "
    "item becomes a real order nobody can fulfil. Prefer the business's own site over aggregators, "
    "and note when a source looks out of date.\n\n"
    "Structure the answer for someone turning it into form fields: group related items, and give "
    "exact names and prices as written."
)


def research_supported(brain: str | None = None) -> bool:
    """Can this request's brain reach the web at all? Callers use it to skip the ask."""
    return (brain or active_brain()) in RESEARCH_BRAINS


async def research(query: str) -> "Research | None":
    """Answer `query` from the live web, or None when this brain cannot search.

    Deliberately NOT part of :func:`generate`. Every brain's structured path pins the output
    shape - Anthropic by forcing a tool, OpenAI via Structured Outputs, Gemini via
    ``response_schema`` - and a pinned shape is exactly what stops a model searching first: under
    a forced ``tool_choice`` the only legal move is to answer. Gemini goes further and rejects a
    search tool and ``response_schema`` in the same call outright.

    So research is its own call, with a search tool and NO schema, and its prose is fed into the
    ordinary build turn as context. The build keeps its guarantee, the research gets the web, and
    neither compromises for the other.

    Never raises. A failed search means the model answers from training data, which is what it did
    before this existed - strictly better than failing a turn the user asked for.
    """
    if not query or not query.strip():
        return None
    brain = active_brain()
    if not research_supported(brain):
        _note_research(brain, "unsupported")
        return None
    cred = _credential()
    if cred is None:
        from .credential import require_credential

        if require_credential():
            return None
        api_key, chosen = _env_key(brain), _default_model(brain)
    else:
        api_key, chosen = cred.api_key, (cred.model or _default_model(brain))

    try:
        if brain == "anthropic":
            found = await _research_anthropic(query, chosen, api_key)
        elif brain == "openai":
            found = await _research_openai(query, chosen, api_key)
        else:
            found = await _research_gemini(query, chosen, api_key)
    except Exception as exc:  # noqa: BLE001 - research is an enhancement, never a failure mode
        log.warning("research: %s search failed, building without it: %s", brain, exc)
        _note_research(brain, "empty")
        return None
    _note_research(brain, "found" if found is not None else "empty")
    return found


def _note_research(brain: str, outcome: str) -> None:
    """Count a research turn. Best-effort: spend visibility must never break the turn itself."""
    try:
        from .metrics import RESEARCH  # lazy import avoids any load-time import cycle

        RESEARCH.labels(brain, outcome).inc()
    except Exception:  # noqa: BLE001
        pass


async def _research_anthropic(query: str, model: str, api_key: str | None) -> "Research | None":
    """Claude with Anthropic's server-side search: it runs the searches and returns cited prose."""
    from anthropic import AsyncAnthropic

    # The SAME cached client as the build turn, with the longer deadline passed PER REQUEST.
    # A separate `anthropic-research` cache key gave every researching workspace TWO entries in a
    # 64-entry LRU, halving how many tenants one worker can hold before it starts evicting and
    # rebuilding TLS + DNS on every turn. The client differed only by its timeout, and the SDK
    # takes that per call.
    client = _cached_client(_client_key("anthropic", api_key),
                            lambda: AsyncAnthropic(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS))
    resp = await client.messages.create(
        timeout=RESEARCH_TIMEOUT_SECONDS,
        model=model,
        max_tokens=RESEARCH_MAX_TOKENS,
        system=RESEARCH_SYSTEM,
        messages=[{"role": "user", "content": query}],
        # No `tool_choice`: the model decides whether this question needs the web at all, so a
        # form that needs no outside facts costs one cheap completion instead of a search.
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": RESEARCH_MAX_USES}],
    )
    text, sources = [], []
    for part in resp.content:
        if getattr(part, "type", None) != "text":
            continue
        text.append(part.text)
        for cite in (getattr(part, "citations", None) or []):
            url = getattr(cite, "url", None)
            if url:
                sources.append({"url": url, "title": getattr(cite, "title", None) or url})
    _u = getattr(resp, "usage", None)
    _note_usage("anthropic", model, getattr(_u, "input_tokens", None), getattr(_u, "output_tokens", None))
    return _finish(" ".join(t.strip() for t in text if t.strip()), sources)


async def _research_openai(query: str, model: str, api_key: str | None) -> "Research | None":
    """OpenAI's ``web_search`` lives on the RESPONSES API, not the chat-completions call the build
    turn uses - chat completions only searches on the dedicated ``*-search-preview`` models. A
    separate research call is what makes that difference invisible to the agent."""
    from openai import AsyncOpenAI

    client = _cached_client(_client_key("openai", api_key),  # shared with the build turn
                            lambda: AsyncOpenAI(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS))
    resp = await client.responses.create(
        timeout=RESEARCH_TIMEOUT_SECONDS,
        model=model,
        instructions=RESEARCH_SYSTEM,
        input=query,
        tools=[{"type": "web_search"}],
        max_output_tokens=RESEARCH_MAX_TOKENS,
        # OpenAI's cap is on the REQUEST, not the tool — the same ceiling as Anthropic's
        # `max_uses`, spelled differently.
        max_tool_calls=RESEARCH_MAX_USES,
    )
    sources = []
    for item in (getattr(resp, "output", None) or []):
        for part in (getattr(item, "content", None) or []):
            for ann in (getattr(part, "annotations", None) or []):
                url = getattr(ann, "url", None)
                if url:
                    sources.append({"url": url, "title": getattr(ann, "title", None) or url})
    _u = getattr(resp, "usage", None)
    _note_usage("openai", model, getattr(_u, "input_tokens", None), getattr(_u, "output_tokens", None))
    return _finish(getattr(resp, "output_text", "") or "", sources)


async def _research_gemini(query: str, model: str, api_key: str | None) -> "Research | None":
    """Gemini grounded on Google Search. NOTE: grounding and ``response_schema`` are mutually
    exclusive on this API, so this call sends no schema at all - the reason research is a separate
    pass rather than a flag on the build turn."""
    from google import genai
    from google.genai import types

    client = _cached_client(_client_key("gemini", api_key), lambda: genai.Client(  # shared
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_SECONDS * 1000)),
    ))
    # `.aio` is the same client's async surface — one cached object serves both paths.
    resp = await client.aio.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=RESEARCH_SYSTEM,
            # No search cap exists on this API (see RESEARCH_MAX_USES); the output ceiling and
            # this deadline are what bound a Gemini research turn. Per request, so the client
            # itself stays shared with the build turn.
            http_options=types.HttpOptions(timeout=int(RESEARCH_TIMEOUT_SECONDS * 1000)),
            max_output_tokens=RESEARCH_MAX_TOKENS,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    sources = []
    for cand in (getattr(resp, "candidates", None) or []):
        meta = getattr(cand, "grounding_metadata", None)
        for chunk in (getattr(meta, "grounding_chunks", None) or []) if meta else []:
            web = getattr(chunk, "web", None)
            url = getattr(web, "uri", None) if web else None
            if url:
                sources.append({"url": url, "title": getattr(web, "title", None) or url})
    _u = getattr(resp, "usage_metadata", None)
    _note_usage("gemini", model, getattr(_u, "prompt_token_count", None),
                getattr(_u, "candidates_token_count", None))
    return _finish(getattr(resp, "text", "") or "", sources)


def _finish(text: str, sources: list[dict]) -> "Research | None":
    """Empty findings are NOT findings. Returning a Research with no text would put an empty
    "here is what I found" section in the build prompt, which reads to the model as "the web says
    nothing about this" rather than "the search did not happen"."""
    if not text.strip():
        return None
    seen, unique = set(), []
    for src in sources:  # a page cited five times is one source to the person reading the form
        if src["url"] not in seen:
            seen.add(src["url"])
            unique.append(src)
    return Research(text=text.strip(), sources=unique)


@lru_cache(maxsize=64)
def _json_schema(response_model: type) -> dict:
    """`model_json_schema()` for a response model, computed once per class.

    Pydantic does NOT memoise it: measured on this service's own `AssistantTurn`, 5.8 ms cold and
    **4.1 ms warm, every call**. That was fine when it happened on a threadpool thread; on the
    event loop it is 4 ms of CPU that no other request can run through, paid on every Anthropic
    turn. The set of response models is small, fixed, and defined at import time, so one entry
    each is all this ever holds.

    Returns the cached dict, so callers must not mutate it — none do; both hand it straight to a
    provider as a tool schema.
    """
    return response_model.model_json_schema()


def _client_key(brain: str, api_key: str | None) -> str:
    """Cache key for a provider SDK client.

    SECURITY: the key material is part of it. Caching on the brain name alone would let a
    client built with one workspace's key serve another workspace's turn — the exact
    cross-tenant leak BYO-key must not have. Hashed, never stored in clear."""
    digest = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:32]
    return f"{brain}:{digest}"


async def _groq(system: str, user: str, response_model: type[T], temperature: float, model: str | None = None,
          *, api_key: str | None = None, allow_fallback: bool = True) -> T:
    from groq import AsyncGroq

    key = api_key if api_key is not None else GROQ_API_KEY
    client = _cached_client(_client_key("groq", key), lambda: AsyncGroq(api_key=key, timeout=LLM_TIMEOUT_SECONDS))  # #25
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
            resp = await client.chat.completions.create(
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


async def _openai(system: str, user: str, response_model: type[T], model: str | None = None,
            *, api_key: str | None = None) -> T:
    """OpenAI GPT-5 nano via Structured Outputs (returns a validated Pydantic model).

    GPT-5 nano is a reasoning model: it rejects `temperature`, so we don't send it.
    `reasoning_effort` is sent only when OPENAI_REASONING_EFFORT is set, and we
    retry without it if the model rejects it (some nano variants 400 on it).
    """
    from openai import AsyncOpenAI

    key = api_key if api_key is not None else OPENAI_API_KEY
    client = _cached_client(_client_key("openai", key),
                            lambda: AsyncOpenAI(api_key=key, timeout=LLM_TIMEOUT_SECONDS))  # #25
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
        completion = await client.beta.chat.completions.parse(**kwargs)
    except Exception:  # noqa: BLE001
        if "reasoning_effort" not in kwargs:
            raise
        kwargs.pop("reasoning_effort")  # model doesn't accept it — retry plainly
        completion = await client.beta.chat.completions.parse(**kwargs)

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


async def _anthropic(system: str, user: str, response_model: type[T], _temperature: float,
               model: str | None = None, *, api_key: str | None = None) -> T:
    """Claude via forced tool use — the provider's way of guaranteeing a schema-shaped result.

    Anthropic has no `response_format`, so we declare the Pydantic schema as a single tool
    and force it with `tool_choice`. The model must answer by "calling" it, and the call's
    input IS the validated object. Costs the schema in input tokens, exactly like OpenAI's
    and Gemini's structured modes.
    """
    from anthropic import AsyncAnthropic

    key = api_key if api_key is not None else ANTHROPIC_API_KEY
    client = _cached_client(_client_key("anthropic", key),
                            lambda: AsyncAnthropic(api_key=key, timeout=LLM_TIMEOUT_SECONDS))  # #25
    chosen = model or ANTHROPIC_MODEL
    tool = {
        "name": "respond",
        "description": "Return the answer in the required shape. You must call this tool.",
        "input_schema": _json_schema(response_model),
    }

    async def call(forced: bool):
        # NOTE: no `temperature`. anthropic 1.x removed it from messages.create and the method
        # takes no **kwargs, so passing it is a TypeError on every turn. Forced tool use already
        # pins the output shape, which is all `temperature` was doing for us on the other brains.
        return await client.messages.create(
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
        resp = await call(forced)
    except Exception as exc:  # noqa: BLE001 - re-raised below unless it is THIS refusal
        if not (forced and _rejects_forced_tool(exc)):
            raise
        log.info("anthropic: %s does not accept a forced tool choice; using auto", chosen)
        _ANTHROPIC_NO_FORCED_TOOL.add(chosen)
        resp = await call(False)

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


async def _gemini(system: str, user: str, response_model: type[T], temperature: float, model: str | None = None,
            *, api_key: str | None = None) -> T:
    from google import genai
    from google.genai import types

    key = api_key if api_key is not None else GEMINI_API_KEY
    client = _cached_client(_client_key("gemini", key), lambda: genai.Client(  # #25
        api_key=key,
        http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_SECONDS * 1000)),  # ms
    ))
    resp = await client.aio.models.generate_content(
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


async def _ollama(system: str, user: str, response_model: type[T], temperature: float, model: str | None = None) -> T:
    import anyio.to_thread
    import ollama

    # The `ollama` package ships no async client, so this one call really is synchronous — and an
    # `async def` wrapping a blocking call is WORSE than the threadpool it replaced: it stalls the
    # whole worker instead of occupying one slot. Hand it to a thread so this brain behaves like
    # the other four from the loop's point of view. (Local dev only; there is no hosted Ollama.)
    def _chat():
        # No key: Ollama is a local daemon, so one cached client for the process is correct.
        return _cached_client("ollama", lambda: ollama.Client(timeout=LLM_TIMEOUT_SECONDS)).chat(  # #25
            model=model or OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=_json_schema(response_model),  # forces schema-shaped JSON
            options={"temperature": temperature},
        )

    resp = await anyio.to_thread.run_sync(_chat)
    _get = resp.get if hasattr(resp, "get") else (lambda k, r=resp: getattr(r, k, None))
    _note_usage("ollama", model or OLLAMA_MODEL, _get("prompt_eval_count"), _get("eval_count"))
    return response_model.model_validate_json(resp["message"]["content"])
