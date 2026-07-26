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
    text = str(exc).lower()
    # Full detail stays server-side only.
    log.warning("brain error (status=%s): %s", status, exc, exc_info=True)
    if status == 429 or "rate limit" in text or "429" in text:
        return HTTPException(status_code=429, detail=busy_message or _BUSY)
    if isinstance(status, int) and 500 <= status < 600:
        # e.g. BrainUnavailable(status_code=503) — preserve the upstream class, generic body.
        return HTTPException(status_code=status, detail=_UNAVAILABLE)
    return HTTPException(status_code=502, detail=_UNAVAILABLE)
