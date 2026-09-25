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
    # By NAME, not by path, and only the endpoints that make a PROVIDER call. `page` serves
    # static HTML; `feedback` does two synchronous psycopg2 writes and no provider call at all.
    # Both are correctly sync: a threadpool slot is exactly what a brief blocking call should
    # take, whereas the same call inside a coroutine stalls the whole worker. Async is for the
    # 85-second wait on someone else's socket, not for everything.
    WORK = {"chat", "testdata", "analyze", "batch", "intake"}
    app = build_app([agent()])
    endpoints = {
        r.endpoint.__name__: r.endpoint
        for r in app.routes
        if getattr(r, "endpoint", None) is not None and r.endpoint.__name__ in WORK
    }
    assert endpoints, "no work endpoints found — the filter has drifted from the routes"
    sync = [n for n, e in endpoints.items() if not inspect.iscoroutinefunction(e)]
    assert not sync, f"these run on a threadpool slot instead of the loop: {sync}"


def test_feedback_stays_synchronous():
    """The other direction, and the one that is easy to get wrong while converting.

    `/feedback` makes no provider call — it writes a label and an audit row through psycopg2,
    synchronously. Converted to `async def` (as it briefly was, because a blanket sweep saw an
    `await` in its body), those writes run ON the loop and stall every other request in the
    worker. A sync endpoint takes one threadpool slot instead, which is what the pool is for.
    """
    app = build_app([FormAgent()])
    fb = next((r.endpoint for r in app.routes
               if getattr(r, "endpoint", None) is not None and r.endpoint.__name__ == "feedback"), None)
    assert fb is not None, "the /feedback route has moved — this guard needs updating"
    assert not inspect.iscoroutinefunction(fb), (
        "/feedback does blocking DB writes; as a coroutine they run on the event loop"
    )


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


def test_the_provider_calls_in_flight_are_bounded():
    """Async removed the bound that the threadpool used to provide by accident.

    A sync endpoint held a pool slot for its whole provider call, so the pool size capped how many
    upstream calls could be open at once whether anyone intended it or not. Awaiting removes that
    entirely: a burst opens as many connections as requests arrive, and the failure lands as the
    provider rate-limiting the workspace's own key, or the box running out of sockets.
    """
    import luke_agents.core.llm as _llm

    calls, peak, current = 0, 0, 0

    async def provider():
        nonlocal calls, peak, current
        calls += 1
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.01)
        current -= 1
        return "ok"

    async def drive():
        return await asyncio.gather(*(_llm._run_brain("test", provider) for _ in range(20)))

    original = _llm.LLM_MAX_INFLIGHT
    try:
        _llm.LLM_MAX_INFLIGHT = 3
        _llm._inflight = None  # rebuild the semaphore at the new size
        out = asyncio.run(drive())
    finally:
        _llm.LLM_MAX_INFLIGHT = original
        _llm._inflight = None

    assert out == ["ok"] * 20, "every turn still completes — the gate queues, it does not drop"
    assert peak <= 3, f"{peak} provider calls were in flight at once, cap was 3"
    assert calls == 20


def test_an_evicted_async_client_is_actually_closed():
    """Their close() returns a COROUTINE. Calling and dropping it closes nothing while raising
    "never awaited" — the sockets leak exactly as they did before the cache was bounded.

    What this proves: the close actually runs. What it does NOT prove is that the `_CLOSING`
    strong reference is load-bearing — I tried, including forcing a collection between the
    eviction and the yield, and could not make an unreferenced task vanish, because CPython's
    loop holds a scheduled task in its ready queue. The reference stays anyway: asyncio's own
    documentation warns that it keeps only a weak one and that callers must hold their own, and
    "I could not reproduce it in a 50ms window" is not evidence that a documented hazard is not
    real."""
    import luke_agents.core.llm as _llm

    closed: list[str] = []

    class AsyncClient:
        def __init__(self, name):
            self.name = name

        async def close(self):
            closed.append(self.name)

    async def drive():
        _llm._clients.clear()
        original = _llm._CLIENT_CACHE_MAX
        try:
            _llm._CLIENT_CACHE_MAX = 1
            _llm._cached_client("a", lambda: AsyncClient("a"))
            _llm._cached_client("b", lambda: AsyncClient("b"))  # evicts "a"
            await asyncio.sleep(0.05)                           # let the close task run
        finally:
            _llm._CLIENT_CACHE_MAX = original
            _llm._clients.clear()

    asyncio.run(drive())
    assert closed == ["a"], f"the evicted client's sockets were never released: {closed}"
