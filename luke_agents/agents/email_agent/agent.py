"""EmailAgent — wires the EmailDoc schema + repair + prompt to the shared core
(LLM brain, rate limiter, server). The user describes an email in natural
language; the LLM returns the FULL EmailDoc each turn; Python repairs/clamps it
so the UI always gets a valid, bounded document, and also generates sample merge
values for preview + test send. Mirrors `form_agent` behind the `Agent` contract.
"""
from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from ...core import Agent, AgentMeta
from ...core import llm
from ...core.ratelimit import enforce
from ...core.tenancy import resolve_tenant
from ...core.transcripts import TurnRecord, safe_record_turn
from .ops import derive_reply, derive_suggestions, extract_variables, repair_doc
from .prompt import SYSTEM, TESTDATA_SYSTEM, build_testdata_message, build_user_message
from .schema import (
    ChatRequest,
    ChatResponse,
    EmailDoc,
    TestDataItem,
    TestDataRequest,
    TestDataResponse,
    TestDataTurn,
)


def _rate_key(request: Request, tenant: str) -> str:
    """Budget key bound to the tenant + originating IP (Render sets X-Forwarded-For).

    Namespaced by tenant (#33) so each org has its own budget, then by IP within the
    tenant. Deliberately NOT the client-supplied ``user_id``: that field is
    unauthenticated and a caller could rotate it per request to mint a fresh budget
    and drive unbounded AI spend. Namespaced by agent slug so each agent has its own
    budget."""
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "anon")
    return f"email:t:{tenant}:ip:{ip}"


class EmailAgent(Agent):
    meta = AgentMeta(
        slug="email",
        name="LukeMail Email Template Builder",
        description="Describe an email in plain language; get a live, bounded EmailDoc back.",
        version="0.1.0",
    )

    def build_router(self) -> APIRouter:
        router = APIRouter(tags=["email"])

        @router.post("/chat", response_model=ChatResponse)
        def chat(req: ChatRequest, request: Request) -> ChatResponse:
            # Auth (require_api_key) is enforced as a router-level dependency in build_app.
            tenant = resolve_tenant(request)
            # Per-tenant + per-IP rate limit FIRST, before any (paid) LLM call.
            enforce(_rate_key(request, tenant))

            # The response_model IS the EmailDoc: json_object mode guarantees a
            # valid document, and we repair/clamp it before returning so the UI
            # always gets a bounded, well-formed doc.
            current_doc = json.dumps(req.doc, default=str) if req.doc else "null"
            user_msg = build_user_message(current_doc, req.message)
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
                doc = llm.generate(SYSTEM, user_msg, EmailDoc, temperature=0.4)
            except Exception as exc:  # invalid JSON, model/network error, rate limit, etc.
                # Record failures too (excluded from training, useful for analysis).
                _record(output=None, changed=None, error=f"{type(exc).__name__}: {exc}")
                status = getattr(exc, "status_code", None)
                text = str(exc).lower()
                if status == 429 or "rate limit" in text or "429" in text:
                    # Shared free/cheap quota is momentarily exhausted — degrade nicely.
                    raise HTTPException(
                        status_code=429,
                        detail="LukeMail is getting a lot of requests right now. "
                        "Please wait a few seconds and try again.",
                    ) from exc
                raise HTTPException(status_code=502, detail=f"brain error: {exc}") from exc

            # Sanitize/repair: drop unknown blocks, clamp bounds, fill theme
            # defaults — so the UI always gets a valid doc.
            doc = repair_doc(doc)
            out_doc = doc.model_dump()
            # Structural compare against the incoming doc: equal means the email is
            # untouched (a question / chit-chat) — UI can skip re-applying.
            current = repair_doc(EmailDoc.model_validate(req.doc)).model_dump() if req.doc else None
            changed = out_doc != current
            title = req.title or doc.subject or "Untitled email"
            # Persist off the response path so it adds no latency to the user's turn.
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

        @router.post("/testdata", response_model=TestDataResponse)
        def testdata(req: TestDataRequest, request: Request) -> TestDataResponse:
            """Generate plausible sample values for each {{var}} in the email, to
            drive the builder's live preview + test send."""
            enforce(_rate_key(request, resolve_tenant(request)))
            try:
                doc = EmailDoc.model_validate(req.doc) if req.doc else EmailDoc()
            except Exception as exc:  # noqa: BLE001 - malformed incoming doc
                raise HTTPException(status_code=422, detail=f"invalid doc: {exc}") from exc
            variables = extract_variables(doc)
            count = max(1, min(req.count or 1, 5))  # cap so one call can't blow the budget
            if not variables:
                # Nothing to fill — return empty sample sets without a paid call.
                return TestDataResponse(samples=[TestDataItem() for _ in range(count)], brain=llm.active_brain())
            try:
                turn = llm.generate(
                    TESTDATA_SYSTEM, build_testdata_message(variables, count), TestDataTurn, temperature=0.6
                )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=502, detail=f"brain error: {exc}") from exc
            samples = turn.samples[:count] or [TestDataItem()]
            return TestDataResponse(samples=samples, brain=llm.active_brain())

        return router
