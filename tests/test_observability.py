"""#21: per-request correlation IDs + structured (JSON) logging."""
import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from luke_agents.core.observability import (
    HEADER,
    CorrelationIdFilter,
    CorrelationIdMiddleware,
    JsonLogFormatter,
    correlation_id_var,
    sanitize_correlation_id,
)


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/cid")
    def cid() -> dict:
        return {"cid": correlation_id_var.get()}  # what the endpoint actually sees

    return app


def test_sanitize_accepts_safe_and_generates_otherwise():
    assert sanitize_correlation_id("abc-123_.X") == "abc-123_.X"
    assert len(sanitize_correlation_id(None)) == 32  # uuid4 hex
    assert len(sanitize_correlation_id("")) == 32
    assert len(sanitize_correlation_id("bad id!")) == 32  # space/!
    assert len(sanitize_correlation_id("x" * 65)) == 32  # too long


def test_inbound_id_reaches_endpoint_and_response():
    client = TestClient(_app())
    r = client.get("/cid", headers={HEADER: "trace-42"})
    assert r.json()["cid"] == "trace-42"          # endpoint saw it (contextvar propagated)
    assert r.headers[HEADER] == "trace-42"         # echoed on the response


def test_generates_id_when_absent():
    client = TestClient(_app())
    r = client.get("/cid")
    assert HEADER in r.headers
    assert r.json()["cid"] == r.headers[HEADER]
    assert len(r.headers[HEADER]) == 32


def test_json_formatter_tags_correlation_id():
    token = correlation_id_var.set("trace-99")
    try:
        record = logging.LogRecord("luke_agents.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
        CorrelationIdFilter().filter(record)
        line = JsonLogFormatter().format(record)
    finally:
        correlation_id_var.reset(token)
    obj = json.loads(line)
    assert obj["correlation_id"] == "trace-99"
    assert obj["message"] == "hello world"
    assert obj["level"] == "INFO"


def test_the_sync_handler_ceiling_is_raised_at_boot():
    """Every agent endpoint is a plain `def`, so anyio's threadpool limit IS this service's
    concurrency ceiling. Its default of 40 was chosen when a turn was one provider call; a
    research turn is three and holds its slot ~3x as long, taking the same 40 slots from ~1.3
    to ~0.47 turns/second. Raising it is nearly free because these handlers sit blocked on a
    provider socket rather than burning CPU.

    The limiter belongs to the running event loop, so this has to be asked from inside one —
    which is also why the app sets it in `lifespan`, on the loop that will serve requests.
    """
    import anyio
    import anyio.to_thread

    from luke_agents.core.server import size_sync_handler_pool

    async def measure() -> int:
        before = anyio.to_thread.current_default_thread_limiter().total_tokens
        size_sync_handler_pool()
        return anyio.to_thread.current_default_thread_limiter().total_tokens, before

    after, before = anyio.run(measure)
    assert before == 40, "anyio's default moved; the reasoning above needs revisiting"
    assert after >= 160
