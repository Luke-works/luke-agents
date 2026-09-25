"""Helpers for testing the async request path without a new test dependency.

`llm.generate` / `llm.research` and every provider call are coroutines now: the agent endpoints
are `async def`, so an 85-second research turn waits on a socket instead of holding one of the
worker's threads. Tests reach them two ways, and both need something here:

  * calling them          -> `run(...)`, a one-line `asyncio.run`;
  * replacing them        -> `returns(...)` / `raises(...)`, which build ASYNC fakes. A plain
                             `lambda: turn` used to work and now fails with "object AssistantTurn
                             can't be used in 'await' expression" — the endpoint awaits it.

anyio's pytest plugin or pytest-asyncio would also do, but neither is a dependency today and a
test-only dep is a poor trade for two helpers.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable


def run(coro):
    """Drive one coroutine to completion, and bridge token usage back to the caller.

    `asyncio.run` executes the coroutine in a COPIED context, so a `ContextVar.set` inside it —
    which is how `_note_usage` records a turn's tokens — is invisible once it returns. Production
    never hits this: the endpoint and `_note_usage` run in the same task context, and
    `last_usage()` is read there. Only a test calling in from sync code sees the copy, so the
    bridge lives here rather than distorting the code under test.
    """
    import luke_agents.core.llm as _llm

    async def _inner():
        out = await coro
        return out, _llm.last_usage()

    out, usage = asyncio.run(_inner())
    if usage is not None:
        _llm._LAST_USAGE.set(usage)
    return out


def returns(value: Any) -> Callable[..., Any]:
    """An async stand-in that answers `value`, whatever it is called with."""

    async def _fake(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return _fake


def record(into: list, value: Any = None) -> Callable[..., Any]:
    """An async stand-in that records its first positional argument, then answers `value`."""

    async def _fake(*args: Any, **_kwargs: Any) -> Any:
        into.append(args[0] if args else None)
        return value

    return _fake


def raises(exc: BaseException) -> Callable[..., Any]:
    """An async stand-in that raises — for the transient/breaker paths."""

    async def _fake(*_args: Any, **_kwargs: Any) -> Any:
        raise exc

    return _fake


def sequence(values: list) -> Callable[..., Any]:
    """An async stand-in that answers each value in turn, repeating the last once exhausted.

    A research turn calls `generate` twice — the ask, then the rebuild with findings — so a test
    that wants to see the second pass has to be able to answer differently.
    """
    remaining = list(values)

    async def _fake(*_args: Any, **_kwargs: Any) -> Any:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return _fake
