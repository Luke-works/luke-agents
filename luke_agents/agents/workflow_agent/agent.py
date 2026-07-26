"""WorkflowAgent — wires the WorkflowDoc schema + repair + prompt to the shared core
(LLM brain, rate limiter, server). The user describes a workflow in natural language;
the LLM returns the FULL WorkflowDoc each turn; Python repairs dangling references so
the visual builder always gets a valid, wireable graph. Mirrors `email_agent` behind
the `Agent` contract.
"""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from ...core import Agent, AgentMeta
from ...core import llm
from ...core.errors import brain_http_error
from ...core.net import client_ip
from ...core.ratelimit import enforce
from ...core.tenancy import resolve_tenant
from ...core.transcripts import TurnRecord, safe_record_turn
from .ops import derive_reply, derive_suggestions, dump_doc, repair_doc
from .prompt import SYSTEM, build_user_message, compact_doc
from .schema import ChatRequest, ChatResponse, WorkflowDocModel


def _rate_key(request: Request, tenant: str) -> str:
    """Budget key bound to the tenant + originating IP (mirrors the email agent):
    namespaced by tenant then IP, never by the unauthenticated client user_id."""
    return f"workflow:t:{tenant}:ip:{client_ip(request)}"


class WorkflowAgent(Agent):
    meta = AgentMeta(
        slug="workflow",
        name="LukeFlow Workflow Builder",
        description="Describe a workflow in plain language; get a valid, wireable WorkflowDoc back.",
        version="0.1.0",
    )

    def build_router(self) -> APIRouter:
        router = APIRouter(tags=["workflow"])

        @router.post("/chat", response_model=ChatResponse)
        def chat(req: ChatRequest, request: Request) -> ChatResponse:
            # Auth (require_api_key) is enforced as a router-level dependency in build_app.
            tenant = resolve_tenant(request)
            # Per-tenant + per-IP rate limit FIRST, before any (paid) LLM call.
            enforce(_rate_key(request, tenant))

            user_msg = build_user_message(compact_doc(req.doc), req.message, req.catalog)
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
            turn_id = str(uuid.uuid4())
            t0 = time.perf_counter()

            def _record(output: dict | None, changed: bool | None, error: str | None) -> None:
                safe_record_turn(TurnRecord(
                    id=turn_id, agent=self.meta.slug,
                    brain=llm.active_brain(), model=llm.active_model(),
                    messages=messages, output=output, input_schema=req.doc,
                    tenant_id=tenant,
                    user_id=req.user_id, session_id=req.session_id, changed=changed,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    error=error, consent=req.consent,
                ))

            try:
                doc = llm.generate(SYSTEM, user_msg, WorkflowDocModel, temperature=0.3)
            except Exception as exc:  # invalid JSON, model/network error, rate limit, etc.
                _record(output=None, changed=None, error=f"{type(exc).__name__}: {exc}")
                status = getattr(exc, "status_code", None)
                text = str(exc).lower()
                if status == 429 or "rate limit" in text or "429" in text:
                    raise HTTPException(
                        status_code=429,
                        detail="LukeFlow is getting a lot of requests right now. "
                        "Please wait a few seconds and try again.",
                    ) from exc
                raise brain_http_error(exc) from exc

            # Repair dangling references so the UI always gets a wireable graph.
            doc = repair_doc(doc)
            out_doc = dump_doc(doc)
            # Structural compare against the incoming doc: equal means the workflow is
            # untouched (a question / chit-chat) — the UI can skip re-applying.
            current = dump_doc(repair_doc(WorkflowDocModel.model_validate(req.doc))) if req.doc else None
            changed = out_doc != current
            title = req.title or doc.name or "Untitled workflow"
            _record(out_doc, changed, None)  # enqueue durably (#41); off-path writer thread
            return ChatResponse(
                doc=out_doc,
                title=title,
                reply=derive_reply(changed, doc),
                suggestions=derive_suggestions(doc),
                changed=changed,
                brain=llm.active_brain(),
                turn_id=turn_id,
            )

        return router
