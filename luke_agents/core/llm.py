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

import os
from typing import TypeVar

from pydantic import BaseModel

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


def generate(system: str, user: str, response_model: type[T], *, temperature: float = 0.3) -> T:
    """Run one turn against the active brain and return a validated `response_model`.

    `system` is the agent's instructions, `user` the per-turn payload. Every
    backend is asked for schema-shaped JSON and the result is validated with
    Pydantic before returning, so callers always get a well-formed object (or an
    exception they can map to an HTTP error).
    """
    brain = active_brain()
    if brain == "groq":
        return _groq(system, user, response_model, temperature)
    if brain == "openai":
        return _openai(system, user, response_model)  # nano ignores temperature
    if brain == "gemini":
        return _gemini(system, user, response_model, temperature)
    return _ollama(system, user, response_model, temperature)


def _groq(system: str, user: str, response_model: type[T], temperature: float) -> T:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
    # The exact output shape lives in the agent's system prompt, so no verbose
    # JSON-schema dump here — keeps input tokens (and cost) down.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # Try the primary model, then fall back to a known-good one on any error
    # (unavailable model id, rate limit, malformed JSON, …).
    models = [GROQ_MODEL]
    if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL:
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
            return response_model.model_validate_json(resp.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - try the next model
            last_err = exc
    raise last_err  # type: ignore[misc]


def _openai(system: str, user: str, response_model: type[T]) -> T:
    """OpenAI GPT-5 nano via Structured Outputs (returns a validated Pydantic model).

    GPT-5 nano is a reasoning model: it rejects `temperature`, so we don't send it.
    `reasoning_effort` is sent only when OPENAI_REASONING_EFFORT is set, and we
    retry without it if the model rejects it (some nano variants 400 on it).
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
    kwargs: dict = {
        "model": OPENAI_MODEL,
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
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        return parsed
    # Fallback (e.g. refusal/edge): validate the raw JSON content ourselves.
    return response_model.model_validate_json(msg.content or "{}")


def _gemini(system: str, user: str, response_model: type[T], temperature: float) -> T:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_SECONDS * 1000)),  # ms
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_model,  # forces schema-shaped JSON
            temperature=temperature,
        ),
    )
    return response_model.model_validate_json(resp.text)


def _ollama(system: str, user: str, response_model: type[T], temperature: float) -> T:
    import ollama

    resp = ollama.Client(timeout=LLM_TIMEOUT_SECONDS).chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=response_model.model_json_schema(),  # forces schema-shaped JSON
        options={"temperature": temperature},
    )
    return response_model.model_validate_json(resp["message"]["content"])
