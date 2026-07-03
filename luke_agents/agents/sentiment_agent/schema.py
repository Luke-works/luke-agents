"""Sentiment agent schema — the LLM contract + request/response models.

The sentiment agent is a STATELESS classifier: given a piece of text (a form
submission, an inbound email, a review, a support message) it returns a small,
BOUNDED judgement — overall sentiment, a confidence score, an urgency hint, and a
few topical themes. The same `SentimentAnalysis` shape is what the LLM emits
(json_object mode) and what the API echoes back, so it must stay self-consistent.

Enum-ish fields are COERCED to an allowed value rather than rejected when the
model emits a near-miss (mirrors the email agent's `_coerce_*` repairs), and
every field is DEFAULTED — so one stray/omitted field never 502s the call, and a
padded batch slot is always valid.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field as PydField, field_validator
from typing_extensions import Annotated

# Bound request size so a single call can't amplify cost or exhaust memory
# (these endpoints are unauthenticated and publicly reachable).
_MAX_TEXT_CHARS = 8_000
# Hard ceiling on items in one /batch payload. The agent ALSO enforces a
# configurable, usually-lower SENTIMENT_BATCH_MAX before the paid call; this is
# just the absolute payload guard.
_MAX_BATCH_ITEMS = 200
_MAX_THEMES = 5
# Email body / document text ceiling. Documents are chunked under this cap; the
# agent's SENTIMENT_DOC_CHUNK_CHARS sets the per-section size.
_MAX_DOC_CHARS = 100_000

Sentiment = Literal["positive", "neutral", "negative"]
Urgency = Literal["low", "medium", "high"]


def _coerce_sentiment(v: object) -> str:
    return v if v in ("positive", "neutral", "negative") else "neutral"


def _coerce_urgency(v: object) -> str:
    return v if v in ("low", "medium", "high") else "low"


class SentimentAnalysis(BaseModel):
    """The judgement the LLM returns for ONE piece of text. Every field is
    defaulted so a stray omission (or a padded batch slot) is still valid."""

    sentiment: Sentiment = PydField(
        default="neutral", description="Overall tone: positive | neutral | negative."
    )
    confidence: float = PydField(
        default=0.0, description="How sure, 0.0 (pure guess) to 1.0 (unambiguous)."
    )
    urgency: Urgency = PydField(
        default="low", description="How soon a human should act: low | medium | high."
    )
    themes: List[str] = PydField(
        default_factory=list,
        description="Up to 5 short lowercase topic tags, e.g. 'billing', 'bug', 'praise'.",
    )
    summary: str = PydField(
        default="", description="One short, neutral sentence explaining the judgement."
    )

    @field_validator("sentiment", mode="before")
    @classmethod
    def _fix_sentiment(cls, v: object) -> str:
        return _coerce_sentiment(v)

    @field_validator("urgency", mode="before")
    @classmethod
    def _fix_urgency(cls, v: object) -> str:
        return _coerce_urgency(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: object) -> float:
        try:
            n = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(n, 1.0))

    @field_validator("themes", mode="before")
    @classmethod
    def _clean_themes(cls, v: object) -> list:
        if not isinstance(v, list):
            return []
        out = [str(t).strip() for t in v if isinstance(t, (str, int, float)) and str(t).strip()]
        return out[:_MAX_THEMES]


class BatchAnalysis(BaseModel):
    """What the LLM returns for a /batch call: one judgement per input text, in
    the SAME ORDER. json_object mode needs an object at the top level, so the list
    is wrapped in `items` — mirrors the email agent's TestDataTurn."""

    items: List[SentimentAnalysis] = PydField(default_factory=list)


class AnalyzeRequest(BaseModel):
    """One piece of text to classify. Stateless — no history needed."""

    text: str = PydField(
        max_length=_MAX_TEXT_CHARS, description="The submission / email / review to analyze."
    )
    user_id: Optional[str] = PydField(default=None, max_length=200)  # advisory only; budget is keyed by IP


class AnalyzeResponse(SentimentAnalysis):
    """Single-text result: the judgement plus which model produced it."""

    brain: str  # which LLM backend ran ("groq" | "openai" | "gemini" | "ollama")
    model: str  # the concrete model id used (e.g. the cheap sentiment model)


class BatchResultItem(SentimentAnalysis):
    """One judgement in a batch, echoing the original text for easy alignment."""

    text: str


class BatchRequest(BaseModel):
    """Many texts in ONE call — far cheaper than N round-trips. The item count is
    additionally capped by SENTIMENT_BATCH_MAX in the agent."""

    texts: List[str] = PydField(description="Texts to analyze, in order.")
    user_id: Optional[str] = PydField(default=None, max_length=200)

    @field_validator("texts")
    @classmethod
    def _bound_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("texts must not be empty")
        if len(v) > _MAX_BATCH_ITEMS:
            raise ValueError(f"too many texts (hard max {_MAX_BATCH_ITEMS})")
        for t in v:
            if len(t) > _MAX_TEXT_CHARS:
                raise ValueError(f"a text exceeds {_MAX_TEXT_CHARS} chars")
        return v


class BatchResponse(BaseModel):
    results: List[BatchResultItem]
    count: int
    brain: str
    model: str


# --------------------------------------------------------------------------- #
# Intake — one source-aware endpoint for forms, emails, and documents.
# Discriminated on `kind` (mirrors the email agent's Block union). The agent
# normalizes each kind to text, applies a source-specific prompt, and (for long
# documents) chunks + aggregates.
# --------------------------------------------------------------------------- #
class IntakeText(BaseModel):
    """Raw text, no special handling — the escape hatch."""

    kind: Literal["text"] = "text"
    text: str = PydField(max_length=_MAX_TEXT_CHARS)


class IntakeForm(BaseModel):
    """A form submission. `data` is the field-key -> value map (matches
    luke-capability-engine's FormInstance.data); `labels` optionally maps those
    keys to human labels (resolved by the caller from the form schema)."""

    kind: Literal["form"] = "form"
    title: Optional[str] = PydField(default=None, max_length=500)
    data: Dict[str, Any] = PydField(description="Submitted answers: field key -> value.")
    labels: Optional[Dict[str, str]] = PydField(
        default=None, description="Optional field key -> human label, to give the model context."
    )


class IntakeEmail(BaseModel):
    """An inbound email. Quoted replies + signatures are stripped server-side
    before classification. (No inbound-email shape is stored in the platform yet;
    this maps cleanly from a Postmark inbound webhook payload.)"""

    kind: Literal["email"] = "email"
    subject: Optional[str] = PydField(default=None, max_length=998)
    body: str = PydField(max_length=_MAX_DOC_CHARS)
    sender: Optional[str] = PydField(default=None, max_length=320)


class IntakeDocument(BaseModel):
    """A document — caller-extracted text (any length up to the cap). Long
    documents are chunked into sections, scored per-section, and rolled up."""

    kind: Literal["document"] = "document"
    title: Optional[str] = PydField(default=None, max_length=500)
    text: str = PydField(max_length=_MAX_DOC_CHARS)


# Discriminated on `kind`: Pydantic routes the payload to the matching model and
# rejects any other kind.
Intake = Annotated[
    Union[IntakeText, IntakeForm, IntakeEmail, IntakeDocument],
    PydField(discriminator="kind"),
]


class IntakeRequest(BaseModel):
    """One intake of any supported kind. Stateless."""

    intake: Intake
    user_id: Optional[str] = PydField(default=None, max_length=200)  # advisory only; budget is keyed by IP


class SectionResult(SentimentAnalysis):
    """One section's judgement within a chunked document, with a preview."""

    index: int
    excerpt: str


class IntakeResponse(SentimentAnalysis):
    """The overall judgement for an intake. For a multi-section document, the
    top-level fields are the AGGREGATE and `sections` carries the per-section
    breakdown (so a human can see where the negative/urgent parts are)."""

    source: str  # which intake kind was analyzed
    brain: str
    model: str
    sections: Optional[List[SectionResult]] = None
