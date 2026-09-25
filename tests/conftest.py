"""Shared test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _fresh_usage_accounting():
    """Start every test with empty token accounting.

    Usage is CUMULATIVE within a request — a LukeBuilds research turn is three provider calls (the
    build that asks for a fact, the search, then the rebuild with the findings) and the transcript
    is the durable per-tenant record of what that cost. It lives in a ContextVar, which outlives
    any one test on the same thread, so without this tests inherit each other's totals.

    That is the same pollution the per-request `llm.reset_usage()` in each agent exists to prevent:
    FastAPI runs sync endpoints on a threadpool whose workers are REUSED, so a turn could
    otherwise be billed for the previous turn on the same thread.
    """
    import luke_agents.core.llm as _llm

    _llm.reset_usage()
    yield
    _llm.reset_usage()
