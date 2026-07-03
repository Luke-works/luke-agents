"""SentimentAgent — a stateless sentiment/intent classifier for short business
text (form submissions, inbound emails, reviews). Mirrors `form_agent` /
`email_agent` behind the shared `Agent` contract: per-tenant + per-IP rate limit,
the shared LLM brain, json_object output validated (and repaired) by Pydantic.

Endpoints:
  * POST /analyze  — one text  -> one judgement.
  * POST /batch    — many texts in ONE LLM call (far cheaper than N calls).

Cost: sentiment is high-volume and simple, so by default this agent runs on a
small/cheap model (SENTIMENT_GROQ_MODEL, default ``llama-3.1-8b-instant``) when
the active brain is Groq — instead of the fleet's heavier GROQ_MODEL — without
changing anything for the form/email agents. Set SENTIMENT_MODEL to pin a
specific model for whichever brain is active.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request

from ...core import Agent, AgentMeta
from ...core import llm
from ...core.ratelimit import enforce
from ...core.tenancy import resolve_tenant
from .intake import aggregate, chunk_text, excerpt, normalize_email, normalize_form
from .prompt import (
    BATCH_SYSTEM,
    DOC_SECTION_SYSTEM,
    INTAKE_SYSTEM,
    SYSTEM,
    build_batch_message,
    build_doc_sections_message,
    build_intake_message,
    build_user_message,
)
from .schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalysis,
    BatchRequest,
    BatchResponse,
    BatchResultItem,
    IntakeRequest,
    IntakeResponse,
    SectionResult,
    SentimentAnalysis,
)

# Explicit per-call model override for ANY active brain. Leave unset to use the
# brain's own default (or the cheap Groq default below).
_SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "").strip() or None
# Cheap/fast Groq model for this agent specifically — keeps high-volume sentiment
# off the fleet's heavier GROQ_MODEL without touching it for other agents.
_SENTIMENT_GROQ_MODEL = os.getenv("SENTIMENT_GROQ_MODEL", "llama-3.1-8b-instant").strip()
# Max texts accepted in one /batch call (still a single LLM call) before the paid request.
_BATCH_MAX = int(os.getenv("SENTIMENT_BATCH_MAX", "50"))
# Per-section size when chunking a long document for /intake. With the schema's
# 100k-char document cap and the default, a document fits in a single batch call.
_DOC_CHUNK_CHARS = int(os.getenv("SENTIMENT_DOC_CHUNK_CHARS", "2000"))
# Classification should be deterministic — no creativity wanted.
_TEMPERATURE = 0.0


def _model_override() -> str | None:
    """Which concrete model this agent should run. Explicit SENTIMENT_MODEL wins
    for any brain; otherwise use the cheap Groq model when Groq is the active
    brain; for other brains fall back to their configured default (None)."""
    if _SENTIMENT_MODEL:
        return _SENTIMENT_MODEL
    if llm.active_brain() == "groq":
        return _SENTIMENT_GROQ_MODEL
    return None


def _rate_key(request: Request, tenant: str) -> str:
    """Budget key bound to tenant + originating IP (Render sets X-Forwarded-For),
    namespaced to this agent. Deliberately NOT keyed by the client-supplied
    user_id, which is unauthenticated and could be rotated to mint fresh budget."""
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "anon")
    return f"sentiment:t:{tenant}:ip:{ip}"


def _map_brain_error(exc: Exception) -> HTTPException:
    """Map an LLM/brain failure to a user-friendly HTTP error (mirrors the other agents)."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status == 429 or "rate limit" in text or "429" in text:
        return HTTPException(
            status_code=429,
            detail="Sentiment analysis is getting a lot of requests right now. "
            "Please wait a few seconds and try again.",
        )
    return HTTPException(status_code=502, detail="agent brain error")


def _analyze_doc_sections(chunks: list[str], model: str | None) -> list[SentimentAnalysis]:
    """Classify document sections in batches (one LLM call per up-to-_BATCH_MAX
    sections), returning one analysis per chunk, aligned by position. Padding fills
    any slot the model omits so every section gets a result."""
    out: list[SentimentAnalysis] = []
    for i in range(0, len(chunks), _BATCH_MAX):
        group = chunks[i : i + _BATCH_MAX]
        try:
            turn = llm.generate(
                DOC_SECTION_SYSTEM,
                build_doc_sections_message(group),
                BatchAnalysis,
                temperature=_TEMPERATURE,
                model=model,
            )
        except Exception as exc:
            raise _map_brain_error(exc) from exc
        items = list(turn.items)[: len(group)]
        while len(items) < len(group):
            items.append(SentimentAnalysis())
        out.extend(items)
    return out


class SentimentAgent(Agent):
    meta = AgentMeta(
        slug="sentiment",
        name="LukeSense Sentiment Analyzer",
        description="Classify the sentiment, urgency, and themes of short business text.",
        version="0.1.0",
    )

    def build_router(self) -> APIRouter:
        router = APIRouter(tags=["sentiment"])

        @router.post("/analyze", response_model=AnalyzeResponse)
        def analyze(req: AnalyzeRequest, request: Request) -> AnalyzeResponse:
            # Auth (require_api_key) is enforced as a router-level dependency in build_app.
            tenant = resolve_tenant(request)
            # Per-tenant + per-IP rate limit FIRST, before the (paid) LLM call.
            enforce(_rate_key(request, tenant))

            model = _model_override()
            try:
                analysis = llm.generate(
                    SYSTEM,
                    build_user_message(req.text),
                    SentimentAnalysis,
                    temperature=_TEMPERATURE,
                    model=model,
                )
            except Exception as exc:  # invalid JSON, model/network error, rate limit, etc.
                raise _map_brain_error(exc) from exc

            return AnalyzeResponse(
                **analysis.model_dump(),
                brain=llm.active_brain(),
                model=model or llm.active_model(),
            )

        @router.post("/batch", response_model=BatchResponse)
        def batch(req: BatchRequest, request: Request) -> BatchResponse:
            tenant = resolve_tenant(request)
            if len(req.texts) > _BATCH_MAX:
                # Reject (don't silently drop) so the caller can chunk and retry.
                raise HTTPException(
                    status_code=413,
                    detail=f"too many texts in one batch (max {_BATCH_MAX}); split into smaller batches.",
                )
            # One LLM call regardless of item count -> one budget unit.
            enforce(_rate_key(request, tenant))

            model = _model_override()
            try:
                turn = llm.generate(
                    BATCH_SYSTEM,
                    build_batch_message(req.texts),
                    BatchAnalysis,
                    temperature=_TEMPERATURE,
                    model=model,
                )
            except Exception as exc:
                raise _map_brain_error(exc) from exc

            # Align results to inputs by position. The model is asked for one item
            # per text in order; if it returns too few, pad with a neutral/unknown
            # default so every input gets a result; if too many, truncate.
            items = list(turn.items)[: len(req.texts)]
            while len(items) < len(req.texts):
                items.append(SentimentAnalysis())
            results = [
                BatchResultItem(**a.model_dump(), text=t)
                for a, t in zip(items, req.texts)
            ]
            return BatchResponse(
                results=results,
                count=len(results),
                brain=llm.active_brain(),
                model=model or llm.active_model(),
            )

        @router.post("/intake", response_model=IntakeResponse)
        def intake(req: IntakeRequest, request: Request) -> IntakeResponse:
            """Source-aware analysis for any intake — a form submission, an inbound
            email, or a document. Forms/emails are normalized then classified in one
            call; long documents are chunked, scored per-section, and rolled up to an
            overall judgement plus a per-section breakdown."""
            tenant = resolve_tenant(request)
            # One LLM call for forms/emails/short docs; long docs do one batch call
            # over their sections (bounded by the doc cap) -> still one budget unit.
            enforce(_rate_key(request, tenant))

            model = _model_override()
            brain = llm.active_brain()
            eff_model = model or llm.active_model()
            p = req.intake
            kind = p.kind

            if kind == "form":
                content = normalize_form(p.data, p.labels, p.title)
            elif kind == "email":
                content = normalize_email(p.subject, p.body, p.sender)
            elif kind == "document":
                content = f"{p.title}\n\n{p.text}" if p.title else (p.text or "")
            else:  # text
                content = p.text or ""

            if not content.strip():
                # Nothing to judge (e.g. an all-empty form) -> neutral, no paid call.
                empty = SentimentAnalysis(summary="No analyzable content in the submission.")
                return IntakeResponse(
                    **empty.model_dump(), source=kind, brain=brain, model=eff_model
                )

            # Documents: chunk + aggregate when longer than one section.
            if kind == "document":
                chunks = chunk_text(content, _DOC_CHUNK_CHARS)
                if len(chunks) > 1:
                    analyses = _analyze_doc_sections(chunks, model)
                    overall = aggregate(analyses)
                    sections = [
                        SectionResult(**a.model_dump(), index=i, excerpt=excerpt(chunks[i]))
                        for i, a in enumerate(analyses)
                    ]
                    return IntakeResponse(
                        **overall.model_dump(),
                        source=kind,
                        brain=brain,
                        model=eff_model,
                        sections=sections,
                    )

            # Single-call path: text, form, email, or a short (one-section) document.
            try:
                analysis = llm.generate(
                    INTAKE_SYSTEM,
                    build_intake_message(kind, content),
                    SentimentAnalysis,
                    temperature=_TEMPERATURE,
                    model=model,
                )
            except Exception as exc:
                raise _map_brain_error(exc) from exc
            return IntakeResponse(
                **analysis.model_dump(), source=kind, brain=brain, model=eff_model
            )

        return router
