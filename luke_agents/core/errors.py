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

# Under bring-your-own-key the quota being hit belongs to the WORKSPACE, not to us. Saying
# "the AI service is busy" sends someone to look at our status page over a limit only they can
# see or raise — and a workspace with a second provider connected has a remedy this wording
# hides. Agents override it with their own product name.
_BUSY = (
    "Your AI provider is rate-limiting this workspace right now. That is your own provider "
    "account's limit rather than ours, so waiting a few seconds usually clears it — or switch "
    "to another connected provider."
)
_UNAVAILABLE = "The AI service is temporarily unavailable. Please try again shortly."
# A timeout is not "unavailable". The provider was reachable and simply did not finish in time,
# and the difference decides what the person does next: "unavailable" sends them to check a status
# page, when the levers they actually have are a faster model and a smaller request. Reported as
# 504 so it is distinguishable in metrics too, rather than hiding inside the 502 bucket.
_TIMED_OUT = ("The model took too long to answer and the request was stopped. Try a faster model, "
              "or ask for less in one go — a large form is quicker built in a few steps.")
_BAD_REQUEST = ("Your AI provider refused this request — usually the chosen model cannot do what "
                "this assistant needs. Pick a different model, or switch provider.")
_REJECTED = ("Your AI provider rejected this workspace's API key. "
             "Reconnect your provider to keep using the assistant.")
_EXHAUSTED = ("Your AI provider account is out of credit or over its quota. "
              "Top it up with your provider, then try again.")

# Machine-stable signal on the response, for core-engine (which holds the key) to act on:
#   missing   -> the workspace has not connected a provider
#   invalid   -> the provider refused the stored key, so it should be marked and re-verified
#   exhausted -> the key is fine but the account is out of credit. NOT a reason to disconnect
#                anyone: the workspace fixes this with their provider, not with us.
# A header rather than a body field so the existing {"detail": "..."} contract is unchanged.
CREDENTIAL_HEADER = "X-AI-Credential"


# Phrases a provider uses to say "this key is not valid". Deliberately specific: they must
# not appear in ordinary prose, because the text we match against can be MODEL OUTPUT (see
# _from_content below).
_KEY_REJECTED_PHRASES = (
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "api key not valid",     # Google returns this with a 400, not a 401
    "api_key_invalid",
    "authentication_error",
)

# A permanently unusable account: the key authenticates fine, but the workspace must go and
# fix their billing. Reported separately from a rate limit, which resolves by waiting.
_ACCOUNT_EXHAUSTED_PHRASES = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing_not_active",
    "credit balance is too low",
)


def _from_content(exc: Exception) -> bool:
    """Whether this exception's text is derived from MODEL OUTPUT rather than the provider's
    own error.

    A pydantic ValidationError embeds the offending input, so a form whose field is labelled
    "Unauthorized access report" used to be enough to classify the turn as a rejected key and
    switch off a workspace whose key was perfectly fine.
    """
    name = type(exc).__name__.lower()
    return "validation" in name or "pydantic" in type(exc).__module__.lower()


def _is_credential_rejection(exc: Exception, status: object) -> bool:
    """Whether the PROVIDER refused the workspace's key, as opposed to failing.

    Deliberately narrow, because a false positive disconnects a working workspace — the worst
    failure mode this feature has. A timeout, a 5xx or a plain 429 says nothing about the key.
    Text is consulted ONLY for a client-error status and never for an exception whose message
    is model output.
    """
    if status in (401, 403):
        return True
    name = type(exc).__name__.lower()
    if "authentication" in name or "permissiondenied" in name:
        return True
    if _from_content(exc):
        return False
    # Google answers a bad key with 400 INVALID_ARGUMENT, so the status alone is not enough —
    # but only trust the text on a client error, and only these exact phrases.
    if status is not None and status not in (400, 404):
        return False
    return any(m in str(exc).lower() for m in _KEY_REJECTED_PHRASES)


def _is_account_exhausted(exc: Exception, status: object) -> bool:
    """Whether the workspace's provider account is out of credit / over quota.

    Separated from a rate limit on purpose: both often arrive as 429, but one clears by
    waiting a few seconds and the other never clears until someone pays. Telling a workspace
    to "wait a few seconds" forever is the difference between a working feature and a
    mystery."""
    if _from_content(exc):
        return False
    return any(m in str(exc).lower() for m in _ACCOUNT_EXHAUSTED_PHRASES)


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
    # Checked before the rate-limit branch too: an exhausted account usually arrives AS a 429.
    if _is_account_exhausted(exc, status):
        return HTTPException(status_code=402, detail=_EXHAUSTED,
                             headers={CREDENTIAL_HEADER: "exhausted"})
    if status == 429 or "rate limit" in text or "429" in text:
        return HTTPException(status_code=429, detail=busy_message or _BUSY)
    # Before the generic fall-through: a timeout carries NO status, so it used to land in the 502
    # "temporarily unavailable" bucket with everything else — which is what a workspace saw after
    # three 30-second attempts against a slow model, with nothing saying so.
    if isinstance(exc, TimeoutError) or any(
        m in text for m in ("timeout", "timed out", "deadline exceeded")
    ):
        return HTTPException(status_code=504, detail=_TIMED_OUT)
    if isinstance(status, int) and 500 <= status < 600:
        # e.g. BrainUnavailable(status_code=503) — preserve the upstream class, generic body.
        return HTTPException(status_code=status, detail=_UNAVAILABLE)
    # A 400 is the provider saying the REQUEST is wrong, which waiting does not fix. Reporting it
    # as "temporarily unavailable; please retry shortly" sent someone into a retry loop against a
    # permanent condition — a workspace on a model that refuses a forced tool choice saw exactly
    # that, every turn, with nothing telling them the model was the problem.
    #
    # The provider's own text is not echoed (it can carry model ids and internal URLs), so this
    # names the actionable part instead: something about this combination is not accepted, and
    # the one lever the person has is the model.
    if status == 400:
        return HTTPException(status_code=422, detail=_BAD_REQUEST)
    return HTTPException(status_code=502, detail=_UNAVAILABLE)
