"""Build the FastAPI app that hosts every agent.

Mounting rules:
  * Each agent's router is mounted under `/agents/<slug>`.
  * An agent with a static UI is served at `/agents/<slug>` and `/agents/<slug>/`.
  * The chosen default agent is ALSO mounted at the root, so a pre-existing
    single-agent client (e.g. luke-consumer-ui hitting `POST /chat`) keeps working
    as a drop-in against this app.
  * `GET /health` reports the active brain and the mounted agents.
  * `GET /` serves the default agent's UI, or a small landing page if it has none.
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse

from .auth import require_api_key
from .llm import active_brain
from .metrics import CONTENT_TYPE as METRICS_CONTENT_TYPE
from .metrics import MetricsMiddleware
from .metrics import render as render_metrics
from .observability import CorrelationIdMiddleware, configure_logging
from .registry import Agent
from .transcripts import _float_env, flush_pending, get_store
from .transcripts import metrics as transcript_metrics

log = logging.getLogger("luke_agents.server")


def _cors_kwargs(raw: str) -> dict:
    """Translate the AGENTS_CORS env into CORSMiddleware kwargs.

    Starlette's `allow_origins` is EXACT-match only — an entry like
    `https://*.lukeflow.com` would never match `https://consdev.lukeflow.com` and
    every preflight from that origin 400s. So we split entries: exact origins go
    to `allow_origins`, and any with a `*` become an `allow_origin_regex` (which
    Starlette matches with fullmatch). `*` alone means allow everything.
    """
    raw = raw.strip()
    if raw == "*":
        return {"allow_origins": ["*"]}
    entries = [o.strip() for o in raw.split(",") if o.strip()]
    exact = [o for o in entries if "*" not in o]
    wild = [o for o in entries if "*" in o]
    kwargs: dict = {}
    if exact:
        kwargs["allow_origins"] = exact
    if wild:
        # Escape each pattern, then turn the wildcard into a DNS-label matcher
        # (e.g. https://*.lukeflow.com -> https://[A-Za-z0-9-]+\.lukeflow\.com).
        kwargs["allow_origin_regex"] = "|".join(
            re.escape(p).replace(r"\*", r"[A-Za-z0-9-]+") for p in wild
        )
    if not kwargs:  # misconfigured (e.g. empty) — fail closed to same-origin only
        kwargs["allow_origins"] = []
    return kwargs


def _mount_static(app: FastAPI, agent: Agent, prefix: str) -> None:
    """Serve the agent's index.html at `prefix` and `prefix + "/"` (no trailing
    slash redirect, so the client's mount-relative fetch works either way)."""
    index = agent.static_index()
    if index is None:
        return

    def page() -> str:
        return index.read_text(encoding="utf-8") if index.exists() else f"<h1>{agent.meta.name}</h1>"

    for path in {prefix or "/", f"{prefix}/"}:
        app.add_api_route(path, page, methods=["GET"], response_class=HTMLResponse, include_in_schema=False)


def assert_prod_hardened() -> None:
    """When AGENTS_ENV marks a production deployment, refuse to start unless the
    security posture is locked down — so a misconfig can't silently ship an open,
    world-CORS, token-burnable service. Mirrors core-engine's strict prod profile.
    No-op unless AGENTS_ENV is prod/production."""
    env = os.getenv("AGENTS_ENV", "").strip().lower()
    if env not in ("prod", "production"):
        return
    problems: list[str] = []
    if not os.getenv("AGENTS_API_KEY", "").strip():
        problems.append("AGENTS_API_KEY unset — endpoints would be unauthenticated")
    cors = os.getenv("AGENTS_CORS", os.getenv("FORM_AGENT_CORS", "*")).strip()
    if cors == "*" or not cors:
        problems.append("AGENTS_CORS is '*'/unset — CORS would be wide open")
    if os.getenv("AGENTS_REQUIRE_TENANT", "").strip().lower() not in ("1", "true", "yes", "on"):
        problems.append("AGENTS_REQUIRE_TENANT not true — all traffic collapses to one budget")
    if problems:
        raise RuntimeError(
            f"Refusing to start in production (AGENTS_ENV={env}): "
            + "; ".join(problems)
            + ". Set these before deploying."
        )


def _install_curated_openapi(app: FastAPI, title: str, api_version: str) -> None:
    """Curate the OpenAPI document (#37): a real description, the X-Agents-Key auth scheme, a
    relative server, and the versioning note — instead of FastAPI's bare default."""

    def custom():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=title,
            version="1.0.0",
            description=(
                f"LukeTalks agent fleet. The canonical API is versioned under `/{api_version}` "
                f"(e.g. `/{api_version}/agents/<slug>/chat`). Legacy unversioned paths "
                "(`/agents/<slug>/...` and the default agent mounted at the root, e.g. `/chat`) "
                "remain for drop-in compatibility and are hidden from this schema. Send the "
                "`X-Agents-Key` header when the server sets `AGENTS_API_KEY`."
            ),
            routes=app.routes,
        )
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["AgentsApiKey"] = {
            "type": "apiKey", "in": "header", "name": "X-Agents-Key",
            "description": "Required when AGENTS_API_KEY is configured on the server.",
        }
        schema["security"] = [{"AgentsApiKey": []}]
        schema["servers"] = [{"url": "/", "description": "this instance"}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom


def build_app(agents: list[Agent], *, default_slug: str | None = None, title: str = "luke-agents") -> FastAPI:
    assert_prod_hardened()  # fail-fast before wiring anything if prod posture is unsafe
    if not agents:
        raise ValueError("build_app needs at least one agent")
    by_slug = {a.meta.slug: a for a in agents}
    if len(by_slug) != len(agents):
        raise ValueError("agent slugs must be unique")
    default = by_slug.get(default_slug) if default_slug else agents[0]

    configure_logging()  # JSON logs tagged with the per-request correlation id (#21)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # #36: modern lifespan (replaces the deprecated @app.on_event start/shutdown hooks).
        # Startup: create the schema/table (or JSONL dir) up front so the first chat doesn't
        # pay for it and a bad DSN surfaces at boot, not mid-request. Never blocks on failure.
        store = get_store()
        try:
            store.init()
            log.info("transcripts backend: %s (ephemeral=%s)", store.name, store.ephemeral)
        except Exception:  # noqa: BLE001
            log.exception("transcripts: init failed (recording will retry per-turn)")
        try:
            yield
        finally:
            # Shutdown: give queued transcript writes a bounded chance to drain before the
            # process exits (Render SIGTERMs on every redeploy). Best-effort (#41).
            try:
                flush_pending(timeout=_float_env("AGENTS_TRANSCRIPT_FLUSH_SECONDS", 5.0))
            except Exception:  # noqa: BLE001
                log.exception("transcripts: flush on shutdown failed")

    # #37: the API is versioned under /v1 (canonical); the app version reflects the API contract,
    # not a stale 0.1.0. Legacy unversioned paths remain for drop-in compatibility.
    api_version = "v1"
    app = FastAPI(title=title, version="1.0.0", lifespan=lifespan)

    # Correlation id first (outermost): added AFTER CORS so it wraps it and every
    # request thread has the id set before any handler/log runs.

    # Allow browser clients (e.g. the consumer-ui Form Builder) to call us. Set
    # AGENTS_CORS to a comma-separated origin list in prod to lock it down;
    # entries may use a `*` subdomain wildcard (e.g. https://*.lukeflow.com).
    origins = os.getenv("AGENTS_CORS", os.getenv("FORM_AGENT_CORS", "*"))
    app.add_middleware(
        CORSMiddleware,
        allow_methods=["*"],
        allow_headers=["*"],
        **_cors_kwargs(origins),
    )
    app.add_middleware(MetricsMiddleware)  # request volume + latency (#22); pure-ASGI, non-buffering
    app.add_middleware(CorrelationIdMiddleware)  # outermost (added last)

    @app.get("/health")
    def health() -> dict:
        # Liveness: "is the process up and serving". Deliberately can't fail on a downstream
        # blip (#35) — Render health-checks this, so a transient DB hiccup must not restart the
        # instance. Readiness (dependency health) is the separate /health/ready probe below.
        store = get_store()
        return {
            "status": "ok",
            "brain": active_brain(),
            "transcripts": store.name,
            "transcripts_ephemeral": store.ephemeral,
            "transcript_writes": transcript_metrics(),
            "default": default.meta.slug,
            "agents": [
                {"slug": a.meta.slug, "name": a.meta.name, "description": a.meta.description,
                 "version": a.meta.version, "path": f"/agents/{a.meta.slug}"}
                for a in agents
            ],
        }

    @app.get("/debug/whoami", include_in_schema=False)
    def _debug_whoami(request: Request) -> dict:
        # TEMPORARY (remove after X-Forwarded-For hop verification). Read-only echo of the
        # forwarding headers so we can count how many reverse-proxy hops Render adds in front of
        # this service — used to set AGENTS_TRUSTED_PROXY_HOPS correctly. No secrets, no side effects.
        xff = request.headers.get("x-forwarded-for")
        hops = [h.strip() for h in xff.split(",")] if xff else []
        return {
            "x_forwarded_for": xff,
            "hops": hops,
            "hop_count": len(hops),
            "direct_peer": request.client.host if request.client else None,
        }

    @app.get("/health/ready")
    def readiness(response: Response) -> dict:
        # Readiness: actually check dependencies (#35). A brain must be resolvable and the
        # transcript store reachable (Postgres SELECT 1; JSONL/null are always ready). Returns
        # 503 when not ready so a load balancer / orchestrator can route around a bad instance
        # without killing it (that's liveness' job).
        store = get_store()
        store_ok = store.ping()
        brain = active_brain()
        brain_ok = bool(brain)
        ready = store_ok and brain_ok
        if not ready:
            response.status_code = 503
        return {"ready": ready, "checks": {"transcripts": store_ok, "brain": brain_ok},
                "brain": brain, "transcripts": store.name}

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        # Prometheus scrape target (#22): request volume + latency + status, plus the transcript
        # write counters. Open (no auth), like /health — it exposes no user data.
        return Response(content=render_metrics(), media_type=METRICS_CONTENT_TYPE)

    # The API-key gate is applied at the router level so it covers every agent
    # route uniformly — including any added later (#32). /health and / are declared
    # on the app above (outside any router) and stay open by design.
    for agent in agents:
        gate = [Depends(require_api_key)]
        # #37: canonical VERSIONED mount (documented in OpenAPI).
        v1_prefix = f"/{api_version}/agents/{agent.meta.slug}"
        app.include_router(agent.build_router(), prefix=v1_prefix, dependencies=gate,
                           tags=[agent.meta.slug])
        _mount_static(app, agent, v1_prefix)
        # Legacy UNVERSIONED mount — kept working for existing clients, hidden from the curated
        # schema so /v1 is the one documented surface (deprecation path).
        legacy_prefix = f"/agents/{agent.meta.slug}"
        app.include_router(agent.build_router(), prefix=legacy_prefix, dependencies=gate,
                           include_in_schema=False)
        _mount_static(app, agent, legacy_prefix)

    # Default agent also at the root for drop-in single-agent compatibility (documented).
    app.include_router(default.build_router(), prefix="", dependencies=[Depends(require_api_key)],
                       tags=[default.meta.slug])

    _install_curated_openapi(app, title, api_version)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        default_index = default.static_index()
        if default_index is not None and default_index.exists():
            return default_index.read_text(encoding="utf-8")
        links = "".join(
            f'<li><a href="/agents/{a.meta.slug}/">{a.meta.name}</a> — {a.meta.description}</li>'
            for a in agents
        )
        return f"<h1>{title}</h1><p>brain: {active_brain()}</p><ul>{links}</ul>"

    return app
