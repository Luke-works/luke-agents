"""The point of the async conversion: a slow turn must not hold a worker thread.

Every agent endpoint used to be a sync `def`, so FastAPI ran it in anyio's threadpool and each
in-flight turn occupied one of its slots for the whole provider call. A LukeBuilds research turn
is three calls — the build that asks for a fact, the web search, then the rebuild with the
findings — bounded at ~85s. Concurrency was capped at the threadpool size, and a burst queued.

Awaiting the provider instead means a turn in flight costs a socket and a coroutine, so these
pin the two properties that make that true, rather than the wall-clock number it produces.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest
from fastapi.testclient import TestClient

import luke_agents.core.llm as llm
from luke_agents.agents.email_agent import EmailAgent
from luke_agents.agents.form_agent import FormAgent
from luke_agents.agents.form_agent.schema import AssistantTurn
from luke_agents.agents.sentiment_agent import SentimentAgent
from luke_agents.agents.workflow_agent import WorkflowAgent
from luke_agents.core.server import build_app


@pytest.mark.parametrize("agent", [FormAgent, EmailAgent, WorkflowAgent, SentimentAgent])
def test_every_endpoint_is_a_coroutine(agent):
    """A sync `def` here silently puts the endpoint back on a threadpool slot — no error, no
    test failure, just the old ceiling returning unnoticed."""
    # By NAME, not by path: `page` serves the agent's static HTML and is correctly sync — it
    # does no provider I/O, so a threadpool slot for the microsecond it takes costs nothing.
    WORK = {"chat", "testdata", "feedback", "analyze", "batch", "intake"}
    app = build_app([agent()])
    endpoints = {
        r.endpoint.__name__: r.endpoint
        for r in app.routes
        if getattr(r, "endpoint", None) is not None and r.endpoint.__name__ in WORK
    }
    assert endpoints, "no work endpoints found — the filter has drifted from the routes"
    sync = [n for n, e in endpoints.items() if not inspect.iscoroutinefunction(e)]
    assert not sync, f"these run on a threadpool slot instead of the loop: {sync}"


def test_the_llm_entry_points_are_awaitable():
    for fn in (llm.generate, llm.research, llm._run_brain,
               llm._groq, llm._openai, llm._anthropic, llm._gemini,
               llm._research_anthropic, llm._research_openai, llm._research_gemini):
        assert inspect.iscoroutinefunction(fn), f"{fn.__name__} still blocks whoever calls it"


def test_a_slow_turn_does_not_block_other_requests(monkeypatch):
    """Requests interleave while one is parked on a provider call.

    Honest about what this does and does not prove: a sync endpoint on a 160-slot threadpool
    would ALSO pass it, because three extra requests fit easily. The test that discriminates
    async from threadpool is `test_every_endpoint_is_a_coroutine` above — this one guards the
    thing that would actually regress in practice, someone reintroducing a BLOCKING call
    (`time.sleep`, a sync SDK, a sync DB read) inside an `async def`, which stalls the whole
    worker and would fail here immediately.
    """
    order: list[str] = []
    gate = asyncio.Event()

    async def fake_generate(system, user, model_cls, **_kw):
        if "SLOW" in user:
            await gate.wait()          # parked on "I/O" — must not stall anyone else
            # (a blocking time.sleep here instead would hang the whole worker, and this test)
            order.append("slow")
        else:
            order.append("fast")
        return AssistantTurn(reply="ok")

    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(llm, "active_brain", lambda: "test")
    monkeypatch.setattr(llm, "active_model", lambda: "test-model")
    monkeypatch.setenv("TRANSCRIPTS_ENABLED", "false")
    monkeypatch.delenv("AGENTS_API_KEY", raising=False)

    app = build_app([FormAgent()])
    schema = {"entities": {}, "root": []}

    async def drive():
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            slow = asyncio.create_task(ac.post("/chat", json={"message": "SLOW", "schema": schema}))
            await asyncio.sleep(0)  # let it reach the await
            fast = [await ac.post("/chat", json={"message": f"quick {i}", "schema": schema})
                    for i in range(3)]
            gate.set()
            slow_res = await slow
            return [r.status_code for r in fast], slow_res.status_code

    fast_codes, slow_code = asyncio.run(drive())

    assert fast_codes == [200, 200, 200]
    assert slow_code == 200
    # Three later requests completed while the first was still parked on its provider call.
    assert order == ["fast", "fast", "fast", "slow"], order
