"""Web research: the pass that lets LukeBuilds build from facts it does not have.

The reported case: "get the menu of Savera Indian Kitchen in Irving Texas and build an order
intake form for take outs". No model knows one restaurant's current menu, and before this the
brain had no web access at all — the only tool we ever sent was the response schema itself,
forced, so the model's single legal move was to answer from training data.

These pin the three things that make the feature safe rather than merely present:
  * research is a SEPARATE call with no schema, so the build turn keeps its forced-tool guarantee;
  * a search that finds nothing produces NO form, never an invented one;
  * it runs at most one extra round, because the bill is the workspace's.
"""
from __future__ import annotations

import pytest

import luke_agents.core.llm as llm


# --------------------------------------------------------------------------- #
# Fakes — one per provider, shaped like the SDK object our code actually reads.
# --------------------------------------------------------------------------- #
class _Obj:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class FakeAnthropic:
    def __init__(self, text="Chicken Biryani $16.99", cite="https://saveraindiankitchen.com/menu"):
        self.calls: list[dict] = []
        self.messages = self
        self._text, self._cite = text, cite

    def create(self, **kw):
        self.calls.append(kw)
        citations = [_Obj(url=self._cite, title="Menu")] if self._cite else []
        return _Obj(content=[_Obj(type="text", text=self._text, citations=citations)],
                    usage=_Obj(input_tokens=10, output_tokens=20))


class FakeOpenAI:
    def __init__(self, text="Chicken Biryani $16.99"):
        self.calls: list[dict] = []
        self.responses = self
        self._text = text

    def create(self, **kw):
        self.calls.append(kw)
        ann = _Obj(url="https://saveraindiankitchen.com/menu", title="Menu")
        item = _Obj(content=[_Obj(annotations=[ann, ann])])  # same page cited twice
        return _Obj(output=[item], output_text=self._text, usage=_Obj(input_tokens=10, output_tokens=20))


class FakeGemini:
    def __init__(self, text="Chicken Biryani $16.99"):
        self.calls: list[dict] = []
        self.models = self
        self._text = text

    def generate_content(self, **kw):
        self.calls.append(kw)
        chunk = _Obj(web=_Obj(uri="https://saveraindiankitchen.com/menu", title="Menu"))
        cand = _Obj(grounding_metadata=_Obj(grounding_chunks=[chunk]))
        return _Obj(text=self._text, candidates=[cand],
                    usage_metadata=_Obj(prompt_token_count=10, candidates_token_count=20))


@pytest.fixture
def platform_key(monkeypatch):
    """No workspace credential, a platform key present — the simplest path through `research`."""
    monkeypatch.setattr(llm, "_credential", lambda: None)
    monkeypatch.setattr(llm, "_env_key", lambda _b: "sk-test")
    monkeypatch.setattr(llm, "_default_model", lambda _b: "test-model")
    monkeypatch.setattr("luke_agents.core.credential.require_credential", lambda: False)


def _use(monkeypatch, brain, fake):
    monkeypatch.setattr(llm, "active_brain", lambda: brain)
    monkeypatch.setattr(llm, "_cached_client", lambda _k, build: fake)


# --------------------------------------------------------------------------- #
# The search actually happens, on each provider that has one
# --------------------------------------------------------------------------- #
def test_anthropic_searches_the_web_and_returns_cited_prose(monkeypatch, platform_key):
    fake = FakeAnthropic()
    _use(monkeypatch, "anthropic", fake)

    found = llm.research("Savera Indian Kitchen Irving Texas takeout menu")

    assert found is not None
    assert "Biryani" in found.text
    assert found.sources == [{"url": "https://saveraindiankitchen.com/menu", "title": "Menu"}]
    sent = fake.calls[0]
    assert sent["tools"] == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": llm.RESEARCH_MAX_USES}
    ]
    # The two properties that make this work AT ALL, and the reason research is its own call:
    # nothing forces a tool (so the model may search first), and no response schema is pinned.
    assert "tool_choice" not in sent
    assert "response_format" not in sent


def test_openai_research_uses_the_responses_api_not_chat_completions(monkeypatch, platform_key):
    # `web_search` does not exist on chat.completions except on the dedicated *-search-preview
    # models, and chat.completions.parse is what the BUILD turn uses. Splitting the call is what
    # keeps that difference out of the agent.
    fake = FakeOpenAI()
    _use(monkeypatch, "openai", fake)

    found = llm.research("Savera Indian Kitchen Irving Texas takeout menu")

    assert found is not None and "Biryani" in found.text
    assert fake.calls[0]["tools"] == [{"type": "web_search"}]
    # One page cited twice is one source to the person reading the form.
    assert found.sources == [{"url": "https://saveraindiankitchen.com/menu", "title": "Menu"}]


def test_gemini_grounds_on_search_and_sends_no_response_schema(monkeypatch, platform_key):
    # Gemini rejects google_search and response_schema in the SAME call, so a grounded build turn
    # is impossible by construction — the strongest reason this is a separate pass.
    fake = FakeGemini()
    _use(monkeypatch, "gemini", fake)

    found = llm.research("Savera Indian Kitchen Irving Texas takeout menu")

    assert found is not None and "Biryani" in found.text
    cfg = fake.calls[0]["config"]
    assert cfg.tools and cfg.tools[0].google_search is not None
    assert getattr(cfg, "response_schema", None) is None
    assert found.sources[0]["url"].endswith("/menu")


# --------------------------------------------------------------------------- #
# Safety: nothing found must mean nothing built
# --------------------------------------------------------------------------- #
def test_the_search_cap_is_sent_wherever_the_provider_has_one(monkeypatch, platform_key):
    """Providers bill per search, on the WORKSPACE's account. The cap has to actually leave the
    process — a constant nobody sends is a comment. Anthropic spells it `max_uses` on the tool,
    OpenAI `max_tool_calls` on the request."""
    fa = FakeAnthropic()
    _use(monkeypatch, "anthropic", fa)
    llm.research("q")
    assert fa.calls[0]["tools"][0]["max_uses"] == llm.RESEARCH_MAX_USES

    fo = FakeOpenAI()
    _use(monkeypatch, "openai", fo)
    llm.research("q")
    assert fo.calls[0]["max_tool_calls"] == llm.RESEARCH_MAX_USES

    # Gemini has no cap to send: types.GoogleSearch exposes no such field. Bounded by the output
    # ceiling and the timeout instead — asserted so the gap is recorded, not merely absent.
    fg = FakeGemini()
    _use(monkeypatch, "gemini", fg)
    llm.research("q")
    cfg = fg.calls[0]["config"]
    assert cfg.max_output_tokens == llm.RESEARCH_MAX_TOKENS
    from google.genai import types as gtypes
    assert "max_uses" not in gtypes.GoogleSearch.model_fields, (
        "google-genai grew a search cap — send it, and update RESEARCH_MAX_USES' comment"
    )


def test_research_shares_the_build_turn_s_client_and_sets_its_deadline_per_request(monkeypatch, platform_key):
    """One SDK client per credential, not two.

    `_cached_client` is a 64-entry LRU holding one httpx client — sockets, TLS context, connection
    pool — per workspace key this worker has served. Giving research its own cache key put TWO
    entries per researching workspace in it, halving how many tenants a worker can hold before it
    evicts and rebuilds TLS + DNS on every turn. The only difference was the timeout, and all
    three SDKs take that per call.
    """
    keys: list[str] = []
    monkeypatch.setattr(llm, "active_brain", lambda: "anthropic")
    fake = FakeAnthropic()

    def spy(key, build):
        keys.append(key)
        return fake

    monkeypatch.setattr(llm, "_cached_client", spy)
    llm.research("Savera Indian Kitchen menu")

    assert keys and not any("research" in k for k in keys), (
        f"research must reuse the build turn's cache key, got {keys}"
    )
    assert keys[0] == llm._client_key("anthropic", "sk-test")
    # The longer deadline still applies — it just travels with the request now.
    assert fake.calls[0]["timeout"] == llm.RESEARCH_TIMEOUT_SECONDS


def test_a_search_that_finds_nothing_returns_nothing(monkeypatch, platform_key):
    # An empty Research would put an empty "here is what I found" block in the build prompt, which
    # reads as "the web says nothing about this" rather than "the search did not happen" — and the
    # model fills that silence with a plausible menu. An invented item becomes a real order.
    _use(monkeypatch, "anthropic", FakeAnthropic(text="   ", cite=None))
    assert llm.research("a restaurant that does not exist") is None


def test_a_provider_error_never_fails_the_turn(monkeypatch, platform_key):
    class Boom:
        messages = property(lambda self: self)

        def create(self, **kw):
            raise RuntimeError("search unavailable")

    _use(monkeypatch, "anthropic", Boom())
    # Degrades to "build from training data", which is exactly what happened before research
    # existed — strictly better than failing a turn the person asked for.
    assert llm.research("anything") is None


def test_brains_without_a_search_tool_are_skipped_not_faked(monkeypatch, platform_key):
    # Groq and Ollama have no first-party search. Giving them one means holding a search vendor's
    # key OURSELVES, which breaks the rule that the work runs on the workspace's account.
    for brain in ("groq", "ollama"):
        monkeypatch.setattr(llm, "active_brain", lambda b=brain: b)
        assert llm.research_supported(brain) is False
        assert llm.research("Savera Indian Kitchen menu") is None
    for brain in ("anthropic", "openai", "gemini"):
        assert llm.research_supported(brain) is True


def test_an_empty_query_never_costs_a_search(monkeypatch, platform_key):
    fake = FakeAnthropic()
    _use(monkeypatch, "anthropic", fake)
    assert llm.research("   ") is None
    assert fake.calls == []  # not merely None — no request was made at all
