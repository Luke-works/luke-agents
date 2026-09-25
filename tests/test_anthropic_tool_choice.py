"""A Claude model that refuses a forced tool choice must still be usable.

Reproduces the live failure of 2026-09-25: every `/agents/form/chat` turn on an Anthropic model
came back 400 —

    tool_choice: type "tool" and "any" are not supported for this model.

— because the brain hard-forced the tool. Forcing it is the strong path and stays the default;
these pin the fallback for the models that reject it.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from tests.aio import raises, record, returns, run, sequence

import luke_agents.core.llm as llm


class Answer(BaseModel):
    reply: str


class _Block:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, content):
        self.content = content
        self.usage = None


class FakeAnthropic:
    """Records how it was called, and refuses a forced choice the way Anthropic does."""

    def __init__(self, *, refuse_forced: bool, answer_as_text: bool = False):
        self.refuse_forced = refuse_forced
        self.answer_as_text = answer_as_text
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kw):
        self.calls.append(kw)
        if self.refuse_forced and kw.get("tool_choice", {}).get("type") in ("tool", "any"):
            err = Exception('tool_choice: type "tool" and "any" are not supported for this model.')
            err.status_code = 400
            raise err
        if self.answer_as_text:
            return _Resp([_Block(type="text", text='Sure! {"reply": "hi"}')])
        return _Resp([_Block(type="tool_use", input={"reply": "hi"})])


@pytest.fixture(autouse=True)
def _forget_learned_models():
    llm._ANTHROPIC_NO_FORCED_TOOL.clear()
    yield
    llm._ANTHROPIC_NO_FORCED_TOOL.clear()


def _use(monkeypatch, fake):
    monkeypatch.setattr(llm, "_cached_client", lambda _k, build: fake)


def test_a_model_that_refuses_a_forced_choice_still_answers(monkeypatch):
    fake = FakeAnthropic(refuse_forced=True)
    _use(monkeypatch, fake)

    out = run(llm._anthropic("sys", "user", Answer, 0.4, model="claude-thinky-1"))

    assert out.reply == "hi"
    # Forced first — it is the only way to guarantee the shape — then auto.
    assert [c["tool_choice"]["type"] for c in fake.calls] == ["tool", "auto"]


def test_the_refusal_is_remembered_so_every_turn_does_not_pay_for_it(monkeypatch):
    fake = FakeAnthropic(refuse_forced=True)
    _use(monkeypatch, fake)

    run(llm._anthropic("sys", "user", Answer, 0.4, model="claude-thinky-1"))
    run(llm._anthropic("sys", "user", Answer, 0.4, model="claude-thinky-1"))

    # Two turns, three calls — not four: the second turn went straight to auto.
    assert [c["tool_choice"]["type"] for c in fake.calls] == ["tool", "auto", "auto"]


def test_a_model_that_accepts_forcing_is_never_downgraded(monkeypatch):
    # The fallback must not become the default. Forced tool use is what guarantees the shape.
    fake = FakeAnthropic(refuse_forced=False)
    _use(monkeypatch, fake)

    run(llm._anthropic("sys", "user", Answer, 0.4, model="claude-haiku-4-5"))

    assert [c["tool_choice"]["type"] for c in fake.calls] == ["tool"]
    assert "claude-haiku-4-5" not in llm._ANTHROPIC_NO_FORCED_TOOL


def test_on_the_auto_path_a_prose_answer_is_still_held_to_the_schema(monkeypatch):
    # With "auto" the model may reply in text instead of calling the tool. Parsing that is a
    # convenience; the schema is still the contract.
    fake = FakeAnthropic(refuse_forced=True, answer_as_text=True)
    _use(monkeypatch, fake)

    out = run(llm._anthropic("sys", "user", Answer, 0.4, model="claude-thinky-1"))
    assert out.reply == "hi"


def test_any_other_400_still_fails_loudly(monkeypatch):
    # A broad match would turn a malformed schema or a bad model name into a silent retry with
    # weaker guarantees — the turn would come back shaped differently and the agent would act on
    # it. Only the tool_choice refusal may fall back.
    class Unrelated(FakeAnthropic):
        async def create(self, **kw):
            self.calls.append(kw)
            err = Exception("model: 'claude-nope' does not exist")
            err.status_code = 400
            raise err

    fake = Unrelated(refuse_forced=True)
    _use(monkeypatch, fake)

    with pytest.raises(Exception, match="does not exist"):
        run(llm._anthropic("sys", "user", Answer, 0.4, model="claude-nope"))
    assert len(fake.calls) == 1, "it must not retry a failure it does not understand"
