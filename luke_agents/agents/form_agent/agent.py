"""FormAgent — wires the form schema + coltorapps mapping + prompt to the shared
core (LLM brain, rate limiter, server). The chat logic that used to live in the
old top-level main.py now lives here, behind the `Agent` contract.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from ...core import Agent, AgentMeta
from ...core import llm
from ...core.ratelimit import enforce
from ...core.transcripts import Feedback, TurnRecord, safe_record_feedback, safe_record_turn
from .coltorapps import schema_to_spec, spec_to_schema
from .prompt import SYSTEM, build_user_message
from .schema import AssistantTurn, ChatRequest, ChatResponse, FeedbackRequest

_STATIC = Path(__file__).parent / "static" / "index.html"


def _rate_key(req: ChatRequest, request: Request) -> str:
    """Identify the caller for rate limiting: the signed-in user id when the
    client sends it, otherwise the originating IP (Render sets X-Forwarded-For).
    Namespaced by agent slug so each agent has its own per-caller budget."""
    if req.user_id:
        who = f"user:{req.user_id}"
    else:
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "anon")
        who = f"ip:{ip}"
    return f"form:{who}"


class FormAgent(Agent):
    meta = AgentMeta(
        slug="form",
        name="LukeTalks Form Builder",
        description="Describe a form in plain language; get a live coltorapps form schema back.",
        version="0.2.0",
    )

    def static_index(self) -> Path:
        return _STATIC

    def build_router(self) -> APIRouter:
        router = APIRouter(tags=["form"])

        @router.post("/chat", response_model=ChatResponse)
        def chat(req: ChatRequest, request: Request, background: BackgroundTasks) -> ChatResponse:
            # Per-user rate limit FIRST, before any (paid) LLM call.
            enforce(_rate_key(req, request))

            # Project the incoming coltorapps schema to a flat spec, keeping the
            # bits we must not lose so the rebuild can merge instead of clobber.
            spec, existing, preserved_entities, preserved_root_ids = schema_to_spec(req.schema)
            if req.title:
                spec.title = req.title

            # The exact messages we send ARE the fine-tuning input — record them as-is.
            user_msg = build_user_message(spec, req.message)
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
            turn_id = str(uuid.uuid4())
            t0 = time.perf_counter()

            def _record(output: dict | None, changed: bool | None, error: str | None) -> None:
                safe_record_turn(TurnRecord(
                    id=turn_id, agent=self.meta.slug,
                    brain=llm.active_brain(), model=llm.active_model(),
                    messages=messages, output=output, input_schema=req.schema,
                    user_id=req.user_id, session_id=req.session_id, changed=changed,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    error=error, consent=req.consent,
                ))

            try:
                turn = llm.generate(SYSTEM, user_msg, AssistantTurn, temperature=0.4)
            except Exception as exc:  # invalid JSON, model/network error, rate limit, etc.
                # Record failures too (excluded from training, useful for analysis).
                _record(output=None, changed=None, error=f"{type(exc).__name__}: {exc}")
                status = getattr(exc, "status_code", None)
                text = str(exc).lower()
                if status == 429 or "rate limit" in text or "429" in text:
                    # Shared free/cheap quota is momentarily exhausted — degrade nicely.
                    raise HTTPException(
                        status_code=429,
                        detail="LukeTalks is getting a lot of requests right now. "
                        "Please wait a few seconds and try again.",
                    ) from exc
                raise HTTPException(status_code=502, detail=f"brain error: {exc}") from exc

            out_schema = spec_to_schema(turn, existing, preserved_entities, preserved_root_ids)
            # Structural compare: equal dicts (any attr order) with equal root order
            # means the form is untouched (a question / chit-chat) — UI can skip re-applying.
            current_schema = req.schema or {"entities": {}, "root": []}
            changed = out_schema != current_schema
            # Persist off the response path so it adds no latency to the user's turn.
            background.add_task(_record, turn.model_dump(), changed, None)
            return ChatResponse(
                schema=out_schema,
                title=turn.title,
                reply=turn.reply,
                suggestions=turn.suggestions,
                changed=changed,
                brain=llm.active_brain(),
                turn_id=turn_id,
            )

        @router.post("/feedback")
        def feedback(req: FeedbackRequest) -> dict:
            """Label a recorded turn (kept/undone, 👍/👎) so the exporter can keep
            only good training examples. Safe no-op if transcripts are disabled."""
            found = safe_record_feedback(
                req.turn_id, Feedback(accepted=req.accepted, rating=req.rating, note=req.note)
            )
            return {"ok": True, "found": found}

        return router
