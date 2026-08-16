"""Tenant resolution for agent requests (#33).

Every paid/recorded request is scoped to a tenant so budgets, transcripts and
exports can be isolated, accounted and erased per organization. The tenant is read
from the ``X-Tenant-Id`` request header — which, in production, the auth gateway
sets from the verified principal after stripping any client-supplied value (the
same pattern core-engine uses). Until that gateway path is the only ingress, treat
the header as the tenant of record and rely on the per-tenant write isolation here.

Default-lenient, on purpose: dev/qa call the agents browser-direct with no gateway
and set no tenant, and a hard failure there would break every local request. So a
missing tenant falls back to ``AGENTS_DEFAULT_TENANT`` (``"public"``). Set
``AGENTS_REQUIRE_TENANT=true`` in an environment that runs behind the gateway to
fail closed (400) when the header is absent.
"""
from __future__ import annotations

import os

from fastapi import HTTPException, Request

TENANT_HEADER = "x-tenant-id"
TIER_HEADER = "x-tenant-tier"
_MAX_LEN = 200

# The commercial plan tiers a request may declare (mirror of core-engine PlanCatalog ids). A tier
# outside this set — or an absent header — resolves to None, and the token budget then falls back to
# its flat cap: agents never has to know the full pricing model, only which bucket to size against.
KNOWN_TIERS = ("FREE", "PRO", "BUSINESS", "ENTERPRISE")


def default_tenant() -> str:
    return (os.getenv("AGENTS_DEFAULT_TENANT", "public").strip() or "public")


def require_tenant() -> bool:
    """When true, reject requests with no tenant instead of using the default."""
    return os.getenv("AGENTS_REQUIRE_TENANT", "").strip().lower() in ("1", "true", "yes", "on")


def resolve_tenant(request: Request) -> str:
    """The tenant this request is scoped to. Never client-trusted for *reads*
    (those go through the ops-controlled export tool); on the write path a spoofed
    value only mislabels the caller's own turn / budget bucket."""
    raw = (request.headers.get(TENANT_HEADER) or "").strip()
    if raw and raw.lower() != "null":
        return raw[:_MAX_LEN]
    if require_tenant():
        raise HTTPException(status_code=400, detail="X-Tenant-Id is required")
    return default_tenant()


def resolve_tier(request: Request) -> str | None:
    """The tenant's commercial plan tier, from the ``X-Tenant-Tier`` header — the gateway/caller
    sets it from the plan core-engine already resolved (``GET /api/plan``), so agents never fetches
    the plan itself. Returns an upper-cased known tier, or ``None`` when the header is absent or not
    a recognized tier. ``None`` is the default-lenient signal: the token budget falls back to its
    flat cap, so a missing/garbage tier can neither block a request nor silently widen its budget."""
    raw = (request.headers.get(TIER_HEADER) or "").strip().upper()
    return raw if raw in KNOWN_TIERS else None
