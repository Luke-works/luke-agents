"""Groq's json_object response_format 400s unless the messages contain the word "json"
(this broke "Generate test data", whose prompt didn't mention json). _groq must inject a
minimal JSON instruction when neither the system nor user prompt already says it."""
import pytest
from pydantic import BaseModel

import luke_agents.core.llm as llm


class _R(BaseModel):
    pass


def _fake_groq(monkeypatch, captured):
    class FakeGroq:
        def __init__(self, **_kwargs):
            pass

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop before validation")  # _groq exhausts models then re-raises

    monkeypatch.setattr("groq.Groq", FakeGroq)
    monkeypatch.setattr(llm, "active_brain", lambda: "groq")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "test-key")


def _messages_blob(captured) -> str:
    return "\n".join(m["content"] for m in captured["messages"]).lower()


def test_injects_json_keyword_when_prompt_omits_it(monkeypatch):
    captured: dict = {}
    _fake_groq(monkeypatch, captured)

    # Neither message says "json" — _groq must add it (mirrors the test-data prompt bug).
    with pytest.raises(RuntimeError):
        llm.generate("Generate distinct test datasets for the form.", "MODE = valid", _R)

    assert captured["response_format"] == {"type": "json_object"}
    assert "json" in _messages_blob(captured)


def test_leaves_prompt_untouched_when_json_already_present(monkeypatch):
    captured: dict = {}
    _fake_groq(monkeypatch, captured)

    with pytest.raises(RuntimeError):
        llm.generate("Output ONLY this JSON object.", "user", _R)

    systems = [m["content"] for m in captured["messages"] if m["role"] == "system"]
    assert systems == ["Output ONLY this JSON object."]  # not modified — already mentions json
