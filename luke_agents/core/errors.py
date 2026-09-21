"""Shared mapping from an LLM/brain failure to a SAFE HTTP error.

Every agent's brain-error path used to `raise HTTPException(502, f"brain error: {exc}")`, which
leaked the raw provider exception (model ids, upstream URLs/messages, quota details) to the client
and flattened the circuit-breaker's `BrainUnavailable` (503) down to a 502. This helper centralizes
the mapping: it logs the full exception server-side and returns a generic, machine-stable body that
never echoes the provider's message.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException

log = logging.getLogger("luke_agents.brain")

_BUSY = (
    "The AI service is receiving a lot of requests right now. "
    "Please wait a few seconds and try again."
)
_UNAVAILABLE = "The AI service is temporarily unavailable. Please try again shortly."
_REJECTED = ("Your AI provider rejected this workspace's API key. "
             "Reconnect your provider to keep using the assistant.")

# Machine-stable signal on the response, for core-engine (which holds the key) to act on:
#   missing -> the workspace has not connected a provider
#   invalid -> the provider refused the stored key, so it should be marked and re-verified
# A header rather than a body field so the existing {"detail": "..."} contract is unchanged.
CREDENTIAL_HEADER = "X-AI-Credential"


def _is_credential_rejection(exc: Exception, status: object) -> bool:
    """Whether the PROVIDER refused the workspace's key, as opposed to failing.

    Deliberately narrow. A timeout, a 5xx or a 429 says nothing about the key, and treating
    one as a rejection would switch off a working workspace because the provider had a bad
    minute — the worst failure mode this feature has."""
    if status in (401, 403):
        return True
    name = type(exc).__name__.lower()
    if "authentication" in name or "permissiondenied" in name:
        return True
    text = str(exc).lower()
    return any(m in text for m in ("invalid api key", "invalid_api_key", "incorrect api key",
                                   "api key not valid", "unauthorized", "authentication_error"))


def brain_http_error(exc: Exception, *, busy_message: str | None = None) -> HTTPException:
    """Map a brain/LLM failure to a client-safe :class:`HTTPException`.

    - honors an explicit ``status_code`` on the exception, so the circuit breaker's
      ``BrainUnavailable`` (503) stays a 503 instead of being flattened to 502;
    - maps rate limits (``status_code == 429`` or a "rate limit"/"429" message) to 429;
    - otherwise returns a generic 502.

    The real cause is logged server-side (with the correlation id already on the log context);
    the returned ``detail`` never contains the provider's raw exception text.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        # Provider SDKs put it on the response, not the exception (e.g. google-genai).
        status = getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc).lower()
    # Full detail stays server-side only.
    log.warning("brain error (status=%s): %s", status, exc, exc_info=True)
    # Checked BEFORE the rate-limit branch: a 401 body can mention quotas, and mislabelling a
    # rejected key as "busy" leaves the workspace retrying forever with no idea what is wrong.
    if _is_credential_rejection(exc, status):
        return HTTPException(status_code=402, detail=_REJECTED,
                             headers={CREDENTIAL_HEADER: "invalid"})
    if status == 429 or "rate limit" in text or "429" in text:
        return HTTPException(status_code=429, detail=busy_message or _BUSY)
    if isinstance(status, int) and 500 <= status < 600:
        # e.g. BrainUnavailable(status_code=503) — preserve the upstream class, generic body.
        return HTTPException(status_code=status, detail=_UNAVAILABLE)
    return HTTPException(status_code=502, detail=_UNAVAILABLE)
