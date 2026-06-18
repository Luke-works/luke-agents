"""Guards the LLM request-timeout fix: the active provider client is constructed
with LLM_TIMEOUT_SECONDS so a hung upstream call can't pin the worker forever."""
import pytest
from pydantic import BaseModel

import luke_agents.core.llm as llm


class _R(BaseModel):
    pass


def test_groq_client_receives_timeout(monkeypatch):
    captured = {}

    class FakeGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        @property
        def chat(self):  # stop before any real network call
            raise RuntimeError("constructed")

    monkeypatch.setattr("groq.Groq", FakeGroq)
    monkeypatch.setattr(llm, "active_brain", lambda: "groq")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "test-key")

    with pytest.raises(RuntimeError):
        llm.generate("system", "user", _R)

    assert captured.get("timeout") == llm.LLM_TIMEOUT_SECONDS
    assert captured.get("api_key") == "test-key"


def test_timeout_default_is_positive():
    assert llm.LLM_TIMEOUT_SECONDS > 0
