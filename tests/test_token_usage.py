"""#D6 — capture LLM token usage: expose it per-turn via last_usage() and count it in Prometheus
(agents_llm_tokens_total) so token spend, the fleet's primary cost, is finally visible."""
from pydantic import BaseModel

import luke_agents.core.llm as llm
from luke_agents.core import metrics as M


class _Reply(BaseModel):
    reply: str = ""


def test_note_usage_sets_last_usage_and_counts_tokens():
    before_p = M.TOKENS.labels("groq", "m1", "prompt")._value.get()
    before_c = M.TOKENS.labels("groq", "m1", "completion")._value.get()

    llm._note_usage("groq", "m1", 10, 4)

    u = llm.last_usage()
    assert u == llm.Usage(prompt_tokens=10, completion_tokens=4)
    assert u.total_tokens == 14
    assert M.TOKENS.labels("groq", "m1", "prompt")._value.get() == before_p + 10
    assert M.TOKENS.labels("groq", "m1", "completion")._value.get() == before_c + 4


def test_note_usage_is_best_effort_on_missing_usage():
    # A provider that omits usage (None) must not raise and must record zero.
    llm._note_usage("groq", "m2", None, None)
    assert llm.last_usage() == llm.Usage(0, 0)


def test_groq_backend_captures_real_response_usage(monkeypatch):
    # A fake Groq response carrying .usage + the parsed content — proves the REAL backend extracts it.
    class _Usage:
        prompt_tokens = 12
        completion_tokens = 8

    class _Msg:
        content = '{"reply":"hi"}'

    class _Choice:
        message = _Msg()

    class _Resp:
        usage = _Usage()
        choices = [_Choice()]

    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**_kw):
                    return _Resp()

    monkeypatch.setattr(llm, "_cached_client", lambda _name, _factory: _Client())
    monkeypatch.setattr(llm, "GROQ_MODEL", "test-model")
    monkeypatch.setattr(llm, "GROQ_FALLBACK_MODEL", "")  # single model, deterministic

    before = M.TOKENS.labels("groq", "test-model", "prompt")._value.get()
    out = llm._groq("system mentions json", "user", _Reply, 0.3)

    assert out.reply == "hi"
    assert llm.last_usage() == llm.Usage(prompt_tokens=12, completion_tokens=8)
    assert M.TOKENS.labels("groq", "test-model", "prompt")._value.get() == before + 12
