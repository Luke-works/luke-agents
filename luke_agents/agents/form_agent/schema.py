"""Form spec schema — the form agent's LLM contract.

The LLM produces a flat, easy-to-get-right field list (`FormSpec`). Python then
deterministically renders that into the coltorapps builder schema that
luke-consumer-ui / luke-capability-engine actually consume (see coltorapps.py).

Field types are the coltorapps palette names so the mapping is loss-free.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field as PydField

# coltorapps palette types we support generating. Choice types (select/radio/
# selectBoxes) require an `options` list.
FieldType = Literal[
    "textField",
    "textarea",
    "number",
    "email",
    "phoneNumber",
    "checkbox",
    "select",
    "radio",
    "selectBoxes",
    "datetime",
    "currency",
    "button",
]

CHOICE_TYPES = {"select", "radio", "selectBoxes"}


class SpecField(BaseModel):
    key: str = PydField(description="Stable snake_case data key, e.g. 'email_address'")
    label: str = PydField(description="Human-facing label")
    type: FieldType = "textField"
    required: bool = False
    options: Optional[List[str]] = PydField(
        default=None, description="Choices for select/radio/selectBoxes; null otherwise"
    )
    placeholder: Optional[str] = PydField(
        default=None, description="Faint example text INSIDE the input; text-like types only"
    )
    tooltip: Optional[str] = PydField(
        default=None, description="Short hover help shown via an info icon next to the label"
    )
    description: Optional[str] = PydField(
        default=None, description="Short helper line shown below the field"
    )


class FormSpec(BaseModel):
    title: str = "Untitled Form"
    fields: List[SpecField] = PydField(default_factory=list)


class FormOp(BaseModel):
    """A single targeted change. The LLM emits ONE op per thing the user asked to
    change — fields it doesn't mention get no op and are left byte-for-byte intact.
    """
    op: Literal["add", "update", "remove", "reorder", "retitle"]
    field: Optional[SpecField] = PydField(
        default=None,
        description="For 'add'/'update': the COMPLETE field after the change. For "
        "'update', keep the SAME `key` as the existing field unless deliberately renaming.",
    )
    after: Optional[str] = PydField(
        default=None, description="For 'add': insert right after this field key (null = append at end)."
    )
    key: Optional[str] = PydField(default=None, description="For 'remove': the field key to delete.")
    order: Optional[List[str]] = PydField(
        default=None, description="For 'reorder': the full list of field keys in the desired order."
    )
    title: Optional[str] = PydField(default=None, description="For 'retitle': the new form title.")


class AssistantTurn(BaseModel):
    """What the LLM returns each turn: a list of targeted operations (EMPTY for a
    question / chit-chat — nothing changes), a conversational reply, and a few
    suggested next steps."""
    operations: List[FormOp] = PydField(
        default_factory=list,
        description="Targeted changes to apply, in order. EMPTY when the message is a "
        "question or general chat (the form must be left untouched).",
    )
    reply: str = PydField(
        default="",
        description="Friendly, first-person natural-language reply describing what you did "
        "(or a clarifying question). 1-3 sentences, conversational.",
    )
    suggestions: List[str] = PydField(
        default_factory=list,
        description="2-4 short, actionable next-step ideas as imperatives, e.g. "
        "'Add a phone number'. Each under ~6 words.",
    )


class TestDataTurn(BaseModel):
    """What the LLM returns when asked to generate test data for a form."""
    values: dict = PydField(
        default_factory=dict,
        description="Map of field key -> a value to enter. For choice fields, use one of "
        "the field's options (a list for multi-select). For 'invalid' mode, values "
        "should deliberately violate the field's rules.",
    )
    notes: str = PydField(
        default="", description="One short line on the data (e.g. which rules the invalid values break)."
    )


class ChatRequest(BaseModel):
    """One turn. Stateless: the client (the Form Builder) sends the current
    coltorapps schema back each time, so the agent needs no storage."""
    message: str
    # Current coltorapps schema: {"entities": {...}, "root": [...]}. Optional /
    # empty for a brand-new form.
    schema: Optional[dict] = None
    title: Optional[str] = None  # current form name, if the client tracks one
    user_id: Optional[str] = None  # for per-user rate limiting (falls back to IP)
    session_id: Optional[str] = None  # stable id grouping turns of one conversation
    consent: bool = True  # may this turn be retained for model fine-tuning?


class ChatResponse(BaseModel):
    schema: dict  # updated coltorapps schema, ready for builderStore / saveDraft
    title: str
    reply: str = ""  # natural-language message to show the user
    suggestions: List[str] = []  # clickable next-step ideas
    changed: bool = True  # False when the form was untouched (e.g. a question)
    brain: str  # which LLM produced this ("groq" | "gemini" | "ollama")
    turn_id: Optional[str] = None  # transcript id; echo to /feedback to label this turn


class FeedbackRequest(BaseModel):
    """Attach a quality label to a recorded turn (by its turn_id), so the
    exporter can keep good examples and drop bad ones."""
    turn_id: str
    accepted: Optional[bool] = None  # user kept (True) or undid (False) the edit
    rating: Optional[int] = None  # +1 / -1 thumbs
    note: Optional[str] = None


class TestDataRequest(BaseModel):
    """Ask LukeTests to generate test data to drive the builder's Test runs."""
    schema: Optional[dict] = None  # current coltorapps schema to generate values for
    mode: Literal["valid", "invalid"] = "valid"  # valid → should pass; invalid → should be rejected
    title: Optional[str] = None
    user_id: Optional[str] = None


class TestDataResponse(BaseModel):
    values: dict  # {field_key: value} to fill into the form
    notes: str = ""
    brain: str
