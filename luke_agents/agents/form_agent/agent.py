"""FormAgent — wires the form schema + coltorapps mapping + prompt to the shared
core (LLM brain, rate limiter, server). The chat logic that used to live in the
old top-level main.py now lives here, behind the `Agent` contract.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from fastapi import APIRouter, HTTPException, Request

from ...core import Agent, AgentMeta
from ...core import llm
from ...core.errors import brain_http_error
from ...core.net import client_ip
from ...core.ratelimit import enforce
from ...core.tenancy import resolve_tenant, resolve_tier
from ...core import tokenbudget
from ...core.observability import correlation_id_var
from ...core.transcripts import (
    AuditRecord, Feedback, TurnRecord, safe_record_audit, safe_record_feedback, safe_record_turn,
)
from .coltorapps import schema_to_spec, spec_to_schema
from .ops import UnsupportedOperation, apply_operations
from .prompt import (OUTBOUND_GUIDANCE, SYSTEM, TESTDATA_SYSTEM, build_research_message,
                     build_testdata_message, build_user_message)
from .schema import (
    AssistantTurn,
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    TestDataItem,
    TestDataRequest,
    TestDataResponse,
    TestDataTurn,
)

_STATIC = Path(__file__).parent / "static" / "index.html"


_BUSY = (
    "Your AI provider is rate-limiting LukeBuilds right now. That is your own provider "
    "account's limit rather than ours, so waiting a few seconds usually clears it — or "
    "switch to another connected provider."
)


def _rate_key(request: Request, tenant: str) -> str:
    """Budget key bound to the tenant + originating IP (Render sets X-Forwarded-For).

    Namespaced by tenant (#33) so each org has its own budget, then by IP within the
    tenant. Deliberately NOT the client-supplied ``user_id``: that field is
    unauthenticated and a caller could rotate it per request to mint a fresh budget
    and drive unbounded AI spend. Namespaced by agent slug so each agent has its own
    budget."""
    return f"form:t:{tenant}:ip:{client_ip(request)}"


class FormAgent(Agent):
    meta = AgentMeta(
        slug="form",
        name="LukeBuilds Form Builder",
        description="Describe a form in plain language; get a live coltorapps form schema back.",
        version="0.2.0",
    )

    def static_index(self) -> Path:
        return _STATIC

    def build_router(self) -> APIRouter:
        router = APIRouter(tags=["form"])

        @router.post("/chat", response_model=ChatResponse)
        async def chat(req: ChatRequest, request: Request) -> ChatResponse:
            # Auth (require_api_key) is enforced as a router-level dependency in build_app.
            tenant = resolve_tenant(request)
            # Per-tenant + per-IP rate limit FIRST, before any (paid) LLM call.
            await run_in_threadpool(enforce, _rate_key(request, tenant))
            # Per-tenant DAILY token cap (D5): reject if this org already hit today's ceiling.
            tokenbudget.enforce(tenant, resolve_tier(request))
            # Threadpool workers are REUSED, and usage is cumulative across a turn's provider
            # calls now, so a turn must not inherit the previous turn's total on the same thread.
            llm.reset_usage()

            # Project the incoming coltorapps schema to a flat spec, keeping the
            # bits we must not lose so the rebuild can merge instead of clobber.
            spec, existing, preserved_entities, preserved_root_ids = schema_to_spec(req.schema)
            if req.title:
                spec.title = req.title

            # The exact messages we send ARE the fine-tuning input — record them as-is.
            # Outbound forms get the two-party guidance appended so the model sets disabled/required
            # field properties that encode who fills what (the fill surface enforces disabled).
            system = SYSTEM + OUTBOUND_GUIDANCE if (req.kind or "").lower() == "outbound" else SYSTEM
            user_msg = build_user_message(spec, req.message)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
            turn_id = str(uuid.uuid4())
            t0 = time.perf_counter()

            def _record(output: dict | None, changed: bool | None, error: str | None) -> None:
                # #64: attribute this turn's tokens (from the LLM layer). Success only — a failed
                # turn's stale/partial usage must not be billed.
                usage = llm.last_usage() if error is None else None
                safe_record_turn(TurnRecord(
                    id=turn_id, agent=self.meta.slug,
                    brain=llm.active_brain(), model=llm.active_model(),
                    messages=messages, output=output, input_schema=req.schema,
                    tenant_id=tenant,
                    user_id=req.user_id, session_id=req.session_id, changed=changed,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    error=error, consent=req.consent,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                ))

            try:
                turn = await llm.generate(system, user_msg, AssistantTurn, temperature=0.4)
            except Exception as exc:  # invalid JSON, model/network error, rate limit, etc.
                # Record failures too (excluded from training, useful for analysis).
                _record(output=None, changed=None, error=f"{type(exc).__name__}: {exc}")
                # Straight to the shared mapper, with this agent's wording. Branching on
                # 429 here first skipped the two checks that deliberately run BEFORE the
                # rate-limit branch in brain_http_error: an out-of-credit account arrives
                # AS a 429, so it was told to "wait a few seconds" forever instead of
                # "top up with your provider" — the exact failure that helper's docstring
                # says it exists to prevent — and the X-AI-Credential header core-engine
                # acts on was dropped with it.
                raise brain_http_error(exc, busy_message=_BUSY) from exc

            # RESEARCH: the model said it needs a fact it does not have, so fetch it and ask
            # again. Bounded to ONE extra round by construction — the second pass is told not to
            # ask, and we never look at `research` on its answer — because a model that can
            # re-ask indefinitely is a loop billed to the workspace.
            sources: list[dict] = []
            research_failed = False
            if turn.research and not turn.action:
                found = await llm.research(turn.research)
                if found is not None:
                    sources = found.sources
                    followup = build_research_message(turn.research, found.text)
                    messages.append({"role": "user", "content": followup})
                    try:
                        turn = await llm.generate(system, f"{user_msg}\n\n{followup}", AssistantTurn,
                                            temperature=0.4)
                    except Exception as exc:  # noqa: BLE001
                        _record(output=None, changed=None, error=f"{type(exc).__name__}: {exc}")
                        raise brain_http_error(exc, busy_message=_BUSY) from exc
                elif llm.research_supported():
                    # Searched and found nothing usable. Say so rather than building a form out
                    # of the model's imagination — the whole point of asking was not knowing.
                    research_failed = True
                    turn.reply = (
                        f"I couldn't find anything usable for \u201c{turn.research}\u201d, so I "
                        "haven't guessed at it. Paste in what you have and I'll build from that."
                    )
                else:
                    research_failed = True
                    turn.reply = (
                        "I'd need to look that up on the web, and this provider can't search. "
                        "Switch to Anthropic, OpenAI or Gemini in the model picker, or paste the "
                        "details in and I'll build from those."
                    )

            # A LIFECYCLE action (check in / publish / undo) is NOT a field edit — ignore any
            # operations the model may have included and leave the form untouched; the app runs
            # the action (and enforces whether it's currently allowed).
            #
            # `research_failed` is the same invariant for the same reason, and it was the sibling
            # this line forgot. A model that asks to look something up may ALSO emit a guess in
            # the same turn; when the lookup then finds nothing, the reply says "I haven't
            # guessed at it" while the guess is applied underneath it. Answering a question the
            # model admitted it could not answer, and contradicting the sentence on screen.
            ops = [] if (turn.action or research_failed) else turn.operations
            # Defense in depth (#26): reject ops referencing unknown op kinds / field types
            # before applying them, even though they already passed Pydantic.
            try:
                new_spec = apply_operations(spec, ops)
            except UnsupportedOperation as exc:
                _record(output=turn.model_dump(), changed=None, error=f"UnsupportedOperation: {exc}")
                raise HTTPException(status_code=502, detail="brain returned an unsupported operation") from exc
            out_schema = spec_to_schema(new_spec, existing, preserved_entities, preserved_root_ids)
            # Structural compare: equal dicts (any attr order) with equal root order
            # means the form is untouched (a question / chit-chat) — UI can skip re-applying.
            current_schema = req.schema or {"entities": {}, "root": []}
            changed = bool(ops) and out_schema != current_schema
            # Enqueue durably (#41): the submit is instant; the write runs off the response
            # path on the transcript-writer thread and is flushed on graceful shutdown.
            _record(turn.model_dump(), changed, None)
            return ChatResponse(
                schema=out_schema,
                title=new_spec.title,
                reply=turn.reply,
                suggestions=turn.suggestions,
                changed=changed,
                action=turn.action,
                brain=llm.active_brain(),
                turn_id=turn_id,
                sources=sources,
            )

        @router.post("/testdata", response_model=TestDataResponse)
        async def testdata(req: TestDataRequest, request: Request) -> TestDataResponse:
            """Generate valid (should pass) or invalid (should be rejected) test data
            for the current form, to drive the builder's Test runs."""
            tenant = resolve_tenant(request)
            await run_in_threadpool(enforce, _rate_key(request, tenant))
            tokenbudget.enforce(tenant, resolve_tier(request))  # per-tenant daily token cap (D5)
            spec, *_ = schema_to_spec(req.schema)
            if req.title:
                spec.title = req.title
            count = max(1, min(req.count or 1, 5))  # cap so one call can't blow the budget
            try:
                turn = await llm.generate(
                    TESTDATA_SYSTEM, build_testdata_message(spec, req.mode, count), TestDataTurn, temperature=0.6
                )
            except Exception as exc:  # noqa: BLE001
                raise brain_http_error(exc) from exc
            datasets = turn.datasets[:count] or [TestDataItem()]
            return TestDataResponse(datasets=datasets, brain=llm.active_brain())

        @router.post("/feedback")
        async def feedback(req: FeedbackRequest, request: Request) -> dict:
            """Label a recorded turn (kept/undone, 👍/👎) so the exporter can keep
            only good training examples. Safe no-op if transcripts are disabled."""
            tenant = resolve_tenant(request)
            await run_in_threadpool(enforce, _rate_key(request, tenant))  # bound writes (was unauthenticated + unthrottled)
            found = safe_record_feedback(
                req.turn_id, Feedback(accepted=req.accepted, rating=req.rating, note=req.note)
            )
            # #40: audit the label change against the VERIFIED principal (the gateway-set
            # tenant), never the client-supplied user_id. Best-effort; never breaks the request.
            safe_record_audit(AuditRecord(
                id=str(uuid.uuid4()), actor=f"tenant:{tenant}", action="feedback.label",
                target=req.turn_id, scope=tenant, request_id=correlation_id_var.get(),
                details={"accepted": req.accepted, "rating": req.rating,
                         "has_note": bool(req.note), "found": found},
            ))
            return {"ok": True, "found": found}

        return router
