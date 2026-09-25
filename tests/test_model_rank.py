"""Ranking models from turns we already ran, rather than from a daily eval nobody paid for."""
from __future__ import annotations

from luke_agents.core.model_rank import MIN_SAMPLES, ModelStat, recommend


def stat(model: str, *, samples: int = 100, ok: float = 0.99,
         p50: int | None = 2000, kept: float | None = 0.9) -> ModelStat:
    return ModelStat(model=model, samples=samples, ok_rate=ok, p50_latency_ms=p50, kept_rate=kept)


def test_a_model_nobody_has_used_enough_is_not_called_proven():
    # Two good turns is not evidence, and a recommendation built on it is worse than none
    # because it looks like one.
    assert not stat("new-one", samples=MIN_SAMPLES - 1).proven
    assert stat("new-one", samples=MIN_SAMPLES).proven


def test_an_unreliable_model_is_never_recommended_however_fast_it_is():
    # Structured output is pass/fail: a model that returns unparseable answers half the time
    # cannot be redeemed by latency.
    quick_but_wrong = stat("fast-liar", ok=0.5, p50=200)
    assert not quick_but_wrong.proven
    picks = recommend([quick_but_wrong], offered=["fast-liar"], provider="groq")
    assert "fast-liar" not in picks


def test_evidence_beats_the_curated_seed_when_they_disagree():
    # The seed is our guess. A measurement that contradicts it must win, or the guess would
    # outlive the thing it was guessing about.
    seeded = "openai/gpt-oss-120b"  # in SEED["groq"]
    bad_news = stat(seeded, ok=0.4)
    picks = recommend([bad_news], offered=[seeded, "llama-3.3-70b-versatile"], provider="groq")
    assert seeded not in picks, "a seed model proven unreliable must not be recommended"
    assert "llama-3.3-70b-versatile" in picks, "the rest of the seed still fills the gap"


def test_the_seed_covers_a_deployment_with_no_history_at_all():
    # The normal state of a new install, and the reason a pure-evidence design would ship
    # nothing useful on day one.
    picks = recommend([], offered=["openai/gpt-oss-120b", "whisper-large-v3"], provider="groq")
    assert picks == ["openai/gpt-oss-120b"]


def test_a_model_this_workspace_cannot_reach_is_never_suggested():
    # Recommendations are per workspace because keys are. Naming a model their account cannot
    # run is worse than saying nothing.
    picks = recommend([stat("claude-haiku-4-5")], offered=["openai/gpt-oss-120b"], provider="groq")
    assert "claude-haiku-4-5" not in picks


def test_ranks_by_reliability_first_and_speed_only_as_a_tie_break():
    reliable_slow = stat("reliable", ok=0.99, p50=6000)
    flakier_fast = stat("flaky", ok=0.91, p50=300)
    picks = recommend([flakier_fast, reliable_slow],
                      offered=["reliable", "flaky"], provider="groq", top_n=2)
    assert picks[0] == "reliable"


def test_no_feedback_is_not_counted_as_rejection():
    # `accepted` is only set when someone actually said. A model used mostly without feedback
    # must not be scored as though people undid its work.
    silent = stat("silent", kept=None)
    disliked = stat("disliked", kept=0.1)
    picks = recommend([silent, disliked], offered=["silent", "disliked"], provider="groq", top_n=2)
    assert picks[0] == "silent"
