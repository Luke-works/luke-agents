"""Shared, agent-agnostic plumbing.

Nothing in here knows about forms (or any specific agent). Agents import these
building blocks; the server wires the agents together.
"""
from .registry import Agent, AgentMeta
from .server import build_app
from .transcripts import (
    Feedback,
    TurnRecord,
    get_store,
    safe_record_feedback,
    safe_record_turn,
)

__all__ = [
    "Agent",
    "AgentMeta",
    "build_app",
    "Feedback",
    "TurnRecord",
    "get_store",
    "safe_record_turn",
    "safe_record_feedback",
]
