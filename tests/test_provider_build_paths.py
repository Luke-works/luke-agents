"""The OpenAI and Gemini BUILD turns, against the async clients they now use.

Neither had a single test. The conversion swapped `OpenAI` for `AsyncOpenAI` and
`client.models.generate_content` for `client.aio.models.generate_content`, and nothing in the
suite would have noticed if either were wrong — the failure would have been a 502 on a real turn,
for a workspace that had brought its own key.

Anthropic and Groq were already covered (test_byo_credential, test_llm_json_object); these close
the gap for the other two.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

import luke_agents.core.llm as llm
from tests.aio import run


class Shaped(BaseModel):
    title: str
    count: int


class _Obj:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _no_platform_key(monkeypatch):
    monkeypatch.setattr(llm, "_credential", lambda: None)
    monkeypatch.setattr("luke_agents.core.credential.require_credential", lambda: False)


def test_openai_build_uses_structured_outputs_on_the_async_client(monkeypatch):
    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            captured["api_key"] = kw.get("api_key")
            self.beta = self

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        async def parse(self, **kw):
            captured.update(kw)
            msg = _Obj(parsed=Shaped(title="Contact form", count=3), content=None)
            return _Obj(choices=[_Obj(message=msg)],
                        usage=_Obj(prompt_tokens=9, completion_tokens=4))

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAI)

    out = run(llm._openai("sys", "usr", Shaped, "gpt-x", api_key="sk-1"))

    assert out == Shaped(title="Contact form", count=3)
    assert captured["api_key"] == "sk-1"
    assert captured["model"] == "gpt-x"
    # Structured Outputs IS the schema guarantee on this brain — the whole reason research had to
    # become a separate call rather than a tool on this one.
    assert captured["response_format"] is Shaped
    assert llm.last_usage().total_tokens == 13


def test_gemini_build_goes_through_the_async_surface_with_a_response_schema(monkeypatch):
    captured: dict = {}

    class FakeModels:
        async def generate_content(self, **kw):
            captured.update(kw)
            return _Obj(text='{"title": "Contact form", "count": 3}',
                        usage_metadata=_Obj(prompt_token_count=5, candidates_token_count=6))

    class FakeClient:
        def __init__(self, **kw):
            captured["api_key"] = kw.get("api_key")
            self.aio = _Obj(models=FakeModels())

    monkeypatch.setattr("google.genai.Client", FakeClient)

    out = run(llm._gemini("sys", "usr", Shaped, 0.3, "gemini-x", api_key="g-1"))

    assert out == Shaped(title="Contact form", count=3)
    assert captured["api_key"] == "g-1"
    # `.aio`, not the sync surface: calling the sync one from a coroutine blocks the whole worker.
    assert captured["model"] == "gemini-x"
    assert captured["config"].response_schema is Shaped
    assert llm.last_usage().total_tokens == 11
