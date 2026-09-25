"""Which models are actually good at a given job, learned from turns we already ran.

A provider lists every model an account can reach — two dozen is normal, and most of them
cannot build a form. Asking someone to choose from that is asking them to guess.

**This costs nothing to produce.** Every turn already records the agent, the model, whether the
model returned something usable, how long it took and what the person did with the result. That
is the same evidence a benchmark would go and buy, so we read it instead of running a daily eval
against every model on somebody else's provider account — under bring-your-own-key that account
belongs to the workspace, and spending their money on work they did not ask for is the one thing
this architecture exists to avoid.

The cost of that choice is honest to state: this measures OUR usage, not a lab. A model nobody
has tried has no evidence, which is what ``SEED`` is for.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# How much history counts. Long enough to gather evidence, short enough that a model which got
# worse — a provider silently repointing an alias is the usual way — stops being recommended.
WINDOW_DAYS = int(os.getenv("MODEL_RANK_WINDOW_DAYS", "14"))

# Below this many turns we do not claim to know anything. Two good turns is not evidence, and a
# recommendation built on it is worse than no recommendation because it looks like one.
MIN_SAMPLES = int(os.getenv("MODEL_RANK_MIN_SAMPLES", "20"))

# A model has to be reliable to be recommended at all. Structured output is pass/fail — the turn
# either parsed into the agent's schema or it did not — so this is a real threshold, not a taste.
MIN_OK_RATE = float(os.getenv("MODEL_RANK_MIN_OK_RATE", "0.9"))

# How many to put in front of someone. Enough to have a choice, few enough to not be a list.
TOP_N = int(os.getenv("MODEL_RANK_TOP_N", "3"))

# ── cold start ────────────────────────────────────────────────────────────────────────────────
# What we suggest before there is evidence. Deliberately small and deliberately marked: these are
# a starting point, not a measurement, and the moment a model has MIN_SAMPLES real turns the
# evidence replaces the guess for that model.
#
# Chosen as each provider's cheapest model that reliably returns structured output, because every
# agent here asks for a schema-shaped answer rather than prose.
SEED: dict[str, list[str]] = {
    "groq": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    "openai": ["gpt-5-nano", "gpt-4.1-mini"],
    "anthropic": ["claude-haiku-4-5", "claude-haiku-4-5-20251001"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-flash"],
}


@dataclass(frozen=True)
class ModelStat:
    """What the turns say about one model doing one agent's job."""

    model: str
    samples: int
    ok_rate: float
    """Turns that produced a usable answer. Structured output makes this pass/fail."""
    p50_latency_ms: int | None
    kept_rate: float | None
    """Of the turns someone acted on, how many they kept rather than undid. None if nobody said."""

    @property
    def proven(self) -> bool:
        return self.samples >= MIN_SAMPLES and self.ok_rate >= MIN_OK_RATE

    def sort_key(self) -> tuple:
        """Reliability, then what people kept, then speed. Best first when sorted descending.

        Banded rather than weighted, and the difference matters. A weighted sum let a model
        failing 9% of turns beat one failing 1%, because it was twenty times faster — arithmetic
        that only makes sense if a failure costs what a slow answer costs. It does not: a failed
        turn is visible to the person, throws away what they typed, and bills them for the
        retry. So reliability decides outright unless two models are genuinely close, and
        "close" is a 5% band rather than a hair, because the numbers below it are noise at the
        sample sizes we have.
        """
        kept = self.kept_rate if self.kept_rate is not None else 0.5  # unknown ≠ bad
        band = round(self.ok_rate * 20)  # 5% buckets
        # Negated so a LOWER latency sorts higher alongside the others.
        return (band, kept, -(self.p50_latency_ms or 10_000))


def recommend(
    stats: list[ModelStat],
    *,
    offered: list[str],
    provider: str,
    top_n: int = TOP_N,
) -> list[str]:
    """The models to put in front of someone, best first.

    ``offered`` is what this workspace's key can actually reach — a recommendation for a model
    they cannot run is worse than none. Evidence wins where it exists; the seed fills the rest,
    which is most of the time on a new deployment.
    """
    can_run = set(offered)
    proven = sorted((s for s in stats if s.proven and s.model in can_run),
                    key=lambda s: s.sort_key(), reverse=True)
    out = [s.model for s in proven[:top_n]]

    # Top up from the seed, skipping anything the evidence has already spoken about — including
    # a seed model that turned out to be unreliable. A curated guess must never override a
    # measurement that disagrees with it.
    judged = {s.model for s in stats if s.samples >= MIN_SAMPLES}
    for candidate in SEED.get(provider, []):
        if len(out) >= top_n:
            break
        if candidate in can_run and candidate not in out and candidate not in judged:
            out.append(candidate)
    return out
