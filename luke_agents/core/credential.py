"""Per-request LLM credential — the workspace brings its own key (BYO-key).

Until now the fleet ran on ONE process-wide provider key read from the environment at
import time (``GROQ_API_KEY`` and friends), so every tenant's turn was billed to Lukeflow
and served by the same client object. The product decision is the same one taken for
Stripe: **Lukeflow enables the capability, it does not pay for it.** Each workspace
connects its own provider account, picks its own model, and gets its own bill.

That turns the provider key into *per-request* state, which has three consequences this
module exists to make safe:

1. **The credential must never be inferred from a client-supplied tenant header.** It is
   handed to us, already resolved, by core-engine — which authenticated the user, checked
   they may act for the tenant, and decrypted the key from ``luke_secrets``. This service
   does not look keys up and cannot be talked into revealing one it was not given.
2. **Nothing derived from a credential may be shared across credentials.** Provider SDK
   clients and circuit-breaker state are cached; cached under a bare brain name
   (``"groq"``) they would hand tenant B the client tenant A's key built. Everything
   credential-derived is therefore keyed by :attr:`Credential.fingerprint`.
3. **The key is secret material in a header.** It is never logged, never echoed in an
   error, never recorded on a transcript, and never returned by ``/health``. Only the
   fingerprint and the last four characters may be surfaced, and only for support.

``AGENTS_REQUIRE_CREDENTIAL=true`` makes the BYO contract mandatory: a turn arriving with
no credential is refused rather than silently falling back to a platform key. That single
flag is also what closes the long-standing hole where an unauthenticated caller could
spend the platform's key — with no platform key in the request path there is nothing to
spend.
"""
from __future__ import annotations

import hashlib
import os
from contextvars import ContextVar
from dataclasses import dataclass

from fastapi import HTTPException, Request

# Headers core-engine sets on a proxied turn. They are trusted only because the gateway
# is the sole ingress in a hardened deployment (AGENTS_API_KEY + non-wildcard CORS, see
# server.assert_prod_hardened); a browser cannot reach this service directly there.
PROVIDER_HEADER = "x-ai-provider"
KEY_HEADER = "x-ai-key"
MODEL_HEADER = "x-ai-model"

# Providers a workspace may bring a key for. "ollama" is deliberately absent: it is a
# local dev backend with no key and no account, reachable only through the env fallback.
SUPPORTED_PROVIDERS = ("groq", "openai", "anthropic", "gemini")

_MAX_KEY_LEN = 512
_MAX_MODEL_LEN = 200


class CredentialRequired(HTTPException):
    """No workspace credential on a turn that requires one.

    402 rather than 401/403 on purpose: the caller is authenticated and allowed, the
    workspace simply has not connected a provider yet. The UI turns this into the
    "Connect your AI provider" prompt instead of an error.
    """

    def __init__(self, detail: str = "This workspace has not connected an AI provider.") -> None:
        # The header is the machine-stable half; core-engine reads it to tell "never connected"
        # from "the provider refused the key" (errors.py sets the same header with "invalid").
        super().__init__(status_code=402, detail=detail, headers={"X-AI-Credential": "missing"})


@dataclass(frozen=True)
class Credential:
    """One workspace's resolved provider credential for the current turn."""

    provider: str
    api_key: str
    model: str | None = None
    #: Where it came from — "tenant" (proxied from core-engine) or "platform" (env
    #: fallback, local dev only). Recorded on transcripts so a turn's cost is attributable.
    source: str = "tenant"

    @property
    def fingerprint(self) -> str:
        """Stable, non-reversible id for this credential — the cache/breaker key.

        Includes the provider so two providers that somehow shared a key string still get
        separate clients, and the model so a per-model client option can never leak across
        models. Truncated to 32 hex chars: collision-free in practice, short in logs.
        """
        material = f"{self.provider}\x00{self.model or ''}\x00{self.api_key}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    @property
    def last_four(self) -> str:
        """Last four characters of the key — the only part safe to show a human."""
        return self.api_key[-4:] if len(self.api_key) >= 4 else ""

    def __repr__(self) -> str:  # pragma: no cover - defensive, but exercised by tests
        """Never let the key reach a log, traceback or error message through repr()."""
        return f"Credential(provider={self.provider!r}, model={self.model!r}, source={self.source!r}, fingerprint={self.fingerprint!r})"

    __str__ = __repr__


# Request-scoped credential, bound by bind() as a router dependency so EVERY agent route
# is covered by default (the argument auth.require_api_key makes). The LLM layer reads it
# without every agent having to thread it through. ContextVars are copied into the
# threadpool FastAPI runs sync endpoints on, so a sync agent handler sees it too.
_CURRENT: ContextVar[Credential | None] = ContextVar("_ai_credential", default=None)


def require_credential() -> bool:
    """When true, a turn with no workspace credential is refused instead of falling back
    to the platform env key. Arm this everywhere the BYO-key contract is real."""
    return os.getenv("AGENTS_REQUIRE_CREDENTIAL", "").strip().lower() in ("1", "true", "yes", "on")


def _clean(raw: str | None, limit: int) -> str:
    """Header values are attacker-influenced even behind the gateway: bound the length and
    strip whitespace/newlines so nothing can be smuggled into a downstream request."""
    return (raw or "").strip()[:limit]


def from_request(request: Request) -> Credential | None:
    """Read the credential core-engine attached to this turn, or None if it sent none."""
    provider = _clean(request.headers.get(PROVIDER_HEADER), 40).lower()
    api_key = _clean(request.headers.get(KEY_HEADER), _MAX_KEY_LEN)
    if not provider and not api_key:
        return None
    if not provider or not api_key:
        # A half-set credential is a caller bug, not a missing credential. Failing loudly
        # beats silently falling back to the platform key and billing the wrong account.
        raise HTTPException(status_code=400, detail="Incomplete AI credential on the request")
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported AI provider: {provider}")
    model = _clean(request.headers.get(MODEL_HEADER), _MAX_MODEL_LEN) or None
    return Credential(provider=provider, api_key=api_key, model=model, source="tenant")


async def bind(request: Request) -> None:
    """FastAPI dependency: resolve this turn's credential and bind it for the LLM layer.

    Refuses the turn when one is required and absent. Applied once in
    ``server.build_app`` so a new agent endpoint is covered automatically rather than
    left on the platform key until someone remembers.

    MUST be ``async``. FastAPI runs a *sync* dependency in a worker thread with a COPY of
    the request context, so a ``ContextVar.set()`` there dies with the copy and the
    endpoint sees nothing — the turn would silently fall back to the platform key. An
    async dependency runs in the request's own task, so the binding is still there when
    the (sync) endpoint is handed a copy of that context.
    """
    cred = from_request(request)
    if cred is None and require_credential():
        raise CredentialRequired()
    _CURRENT.set(cred)


def current() -> Credential | None:
    """This turn's workspace credential, or None to use the env/platform fallback."""
    return _CURRENT.get()


def set_current(cred: Credential | None) -> object:
    """Bind a credential directly (tests, and non-HTTP callers like the CLI tools).
    Returns the ContextVar token so the caller can reset()."""
    return _CURRENT.set(cred)


def reset(token: object) -> None:
    """Undo a set_current(), restoring the previous binding."""
    _CURRENT.reset(token)  # type: ignore[arg-type]
