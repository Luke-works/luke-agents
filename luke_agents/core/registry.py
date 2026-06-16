"""The Agent contract.

An agent is a self-contained unit: some metadata, a FastAPI router (its
endpoints), and optionally a static UI to serve. The server mounts each agent
under `/agents/<slug>` and, for the chosen default, also at the root so existing
single-agent clients keep working. Adding a new agent = subclass `Agent`,
implement `build_router`, and pass an instance to `build_app`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter


@dataclass(frozen=True)
class AgentMeta:
    slug: str  # url-safe segment, e.g. "form" -> mounted at /agents/form
    name: str  # human label, e.g. "LukeBuilds Form Builder"
    description: str  # one line shown on the landing page / health
    version: str = "0.1.0"


class Agent:
    """Base class for an agent. Subclasses set `meta` and implement `build_router`."""

    meta: AgentMeta

    def build_router(self) -> APIRouter:
        """Return the agent's endpoints. Mounted under `/agents/<slug>`."""
        raise NotImplementedError

    def static_index(self) -> Path | None:
        """Optional path to an `index.html` test client served at the agent root.
        Return None for a headless (API-only) agent."""
        return None
