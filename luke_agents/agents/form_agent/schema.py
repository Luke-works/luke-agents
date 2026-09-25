"""Form spec schema — the form agent's LLM contract.

The LLM produces a flat, easy-to-get-right field list (`FormSpec`). Python then
deterministically renders that into the coltorapps builder schema that
luke-consumer-ui / luke-capability-engine actually consume (see coltorapps.py).

Field types are the coltorapps palette names so the mapping is loss-free.
"""
from __future__ import annotations

import json
from typing import List, Literal, Optional

from pydantic import BaseModel, Field as PydField, field_validator

# Bound request size so a single call can't amplify cost or exhaust memory
# (these endpoints are unauthenticated and publicly reachable).
_MAX_MESSAGE_CHARS = 16_000
_MAX_SCHEMA_BYTES = 200_000


def _validate_schema_size(v: Optional[dict]) -> Optional[dict]:
    if v is not None and len(json.dumps(v, default=str)) > _MAX_SCHEMA_BYTES:
        raise ValueError("schema is too large")
    return v

# coltorapps palette types we support generating. Choice types (select/radio/
# selectBoxes) require an `options` list.
#: Every type the agent may emit. Held to form-core's registry by tests/test_field_type_parity.py,
#: which fails when the builder grows a type this list has not accounted for — the drift that let
#: the agent tell a user "the builder only has basic field types" while they looked at a palette
#: containing tabs, panels and a stepper.
FieldType = Literal[
    # ── text and contact ────────────────────────────────────────────────────
    "textField",
    "textarea",
    "richText",
    "email",
    "phoneNumber",
    "url",
    "password",
    "addressBlock",
    # ── numbers, money, quantities ──────────────────────────────────────────
    "number",
    "currency",
    "stepper",
    "rating",
    # ── choices ─────────────────────────────────────────────────────────────
    "checkbox",
    "select",
    "searchSelect",
    "radio",
    "selectBoxes",
    "tags",
    "ranking",
    "matrix",
    # ── dates and times ─────────────────────────────────────────────────────
    "datetime",
    "day",
    "time",
    # ── uploads and marks ───────────────────────────────────────────────────
    "file",
    "signature",
    # ── layout containers (hold `children`; see CONTAINER_TYPES) ────────────
    "panel",
    "fieldset",
    "well",
    "columns",
    "tabs",
    "table",
    "wizard",
    "page",
    # ── static: no answer, just structure on the page ───────────────────────
    "heading",
    "content",
    "divider",
    "button",
]

CHOICE_TYPES = {"select", "searchSelect", "radio", "selectBoxes", "tags", "ranking"}

#: Types that HOLD other fields. Only these may carry `children`, and they hold no value of their
#: own. Kept in step with form-core's `isContainer` by the parity test: a container the agent
#: treated as a leaf would silently drop everything inside it.
CONTAINER_TYPES = {"panel", "fieldset", "well", "columns", "tabs", "table", "wizard", "page"}

#: Types that render but collect nothing — a heading, a rule, a paragraph, the submit button.
#: They take no `required`, no `key` that matters, and never appear in a submission.
STATIC_TYPES = {"heading", "content", "divider", "button"}

#: Containers whose children must themselves be containers, because each child IS a section and
#: its label names that section. Verified against form-core, not assumed:
#:   * `tabs`  -> `panel` children; the panel's label is the tab's label
#:     (form-builder/src/palette.tsx seeds exactly two panels when you drop a Tabs).
#:   * `wizard` -> `page` children; `wizardPages` in form-react filters children to `page` and a
#:     wizard with none is not treated as a wizard at all (helpers.ts:166).
#: Put a bare field straight under either and you get a tab strip with no tabs, or a wizard that
#: silently renders as an ordinary form.
SECTIONED_CONTAINERS = {"tabs": "panel", "wizard": "page"}


class SpecLogicRule(BaseModel):
    """One conditional rule on a field: when `when` evaluates truthy, apply `action`.
    Expressions reference OTHER fields by their key (e.g. name == "gowtham", age >= 18)."""
    when: str = PydField(
        default="",
        description="Boolean expression gating the rule (empty = always fires). Reference other "
        'fields by key. Operators: == != > < >= <= and or not + - * /. String literals use double '
        'quotes, e.g. name == "gowtham". NOT JavaScript — just this small expression grammar.',
    )
    action: Literal["show", "hide", "enable", "disable", "require", "optional", "setValue"] = PydField(
        description="What to do when `when` is truthy: show / hide (conditional visibility), "
        "enable / disable, require / optional, or setValue."
    )
    value: Optional[str] = PydField(
        default=None, description="For action='setValue' ONLY: an expression for the value to set."
    )


class SpecField(BaseModel):
    key: str = PydField(description="Stable snake_case data key, e.g. 'email_address'")
    label: str = PydField(description="Human-facing label")
    type: FieldType = "textField"
    required: bool = False
    options: Optional[List[str]] = PydField(
        default=None, description="Choices for select/radio/selectBoxes; null otherwise"
    )
    children: Optional[List["SpecField"]] = PydField(
        default=None,
        description="The fields INSIDE a layout container (panel, fieldset, well, columns, tabs, "
        "table, wizard, page). Null for every ordinary field. A container holds no answer of its "
        "own — it groups. Tabs take `panel` children (each panel's label is its tab); a wizard "
        "takes `page` children (each page is a step). Nest only as deep as the form needs.",
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
    # ── Advanced (omit / null unless asked). These are real form behaviours the builder applies. ──
    hidden: Optional[bool] = PydField(
        default=None, description="Always hide this field (it's still submitted). Use `logic` for CONDITIONAL hiding."
    )
    disabled: Optional[bool] = PydField(default=None, description="Render the field non-editable (disabled).")
    logic: Optional[List[SpecLogicRule]] = PydField(
        default=None,
        description="Conditional rules — the way to show / hide / require a field based on OTHER fields. "
        'E.g. hide this field when Name is "gowtham": [{"when": "name == \\"gowtham\\"", "action": "hide"}].',
    )
    calculate_value: Optional[str] = PydField(
        default=None,
        description="Expression that auto-computes this field's value from other fields, e.g. "
        "quantity * unit_price. Recomputed whenever a referenced field changes.",
    )


SpecField.model_rebuild()  # `children` refers to SpecField itself


class FormSpec(BaseModel):
    title: str = "Untitled Form"
    fields: List[SpecField] = PydField(default_factory=list)


class FormOp(BaseModel):
    """A single targeted change. The LLM emits ONE op per thing the user asked to
    change — fields it doesn't mention get no op and are left byte-for-byte intact.
    """
    op: Literal["add", "update", "remove", "reorder", "retitle"]
    parent: Optional[str] = PydField(
        default=None,
        description="For 'add'/'reorder': the KEY of the layout container to act inside (a panel, "
        "a tab's panel, a wizard page). Null means the top level of the form. Only meaningful "
        "when that key names a container; 'update' and 'remove' find their target wherever it is "
        "and need no parent.",
    )
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
    research: Optional[str] = PydField(
        default=None,
        description="Set ONLY when building the form correctly needs a fact you do not have and "
        "cannot reason out: a named real-world business's menu or price list, a published "
        "standard's required fields, current rates. Put the SEARCH QUERY here (e.g. 'Savera "
        "Indian Kitchen Irving Texas takeout menu with prices'), leave `operations` EMPTY, and "
        "you will be called again with what was found. Do NOT set it to look up something you "
        "already know, and never guess the facts instead of asking. Null for everything else.",
    )
    action: Optional[Literal["checkin", "publish", "undo_checkout"]] = PydField(
        default=None,
        description="Set ONLY when the user asks to perform a LIFECYCLE action on the whole form "
        "rather than edit fields: 'checkin' (check in / commit / save a version / snapshot), "
        "'publish' (publish / go live / make it live), or 'undo_checkout' (undo checkout / roll "
        "back / discard changes / revert). When set, leave `operations` EMPTY. Null for everything else.",
    )


class TestDataItem(BaseModel):
    """One generated dataset."""
    values: dict = PydField(
        default_factory=dict,
        description="Map of field key -> a value to enter. For choice fields, use one of "
        "the field's options (a list for multi-select). For 'invalid' mode, values "
        "should deliberately violate the field's rules.",
    )
    notes: str = PydField(
        default="", description="One short line on this dataset (the persona, or which rules the invalid values break)."
    )


class TestDataTurn(BaseModel):
    """What the LLM returns when asked to generate test data for a form."""
    datasets: List[TestDataItem] = PydField(
        default_factory=list,
        description="COUNT distinct datasets — meaningfully different (different realistic "
        "personas for valid mode; different broken rules for invalid mode).",
    )


class ChatRequest(BaseModel):
    """One turn. Stateless: the client (the Form Builder) sends the current
    coltorapps schema back each time, so the agent needs no storage."""
    message: str = PydField(max_length=_MAX_MESSAGE_CHARS)
    # Current coltorapps schema: {"entities": {...}, "root": [...]}. Optional /
    # empty for a brand-new form.
    schema: Optional[dict] = None
    title: Optional[str] = PydField(default=None, max_length=500)
    # Authoring intent: "outbound" switches on two-party guidance (disabled display fields vs
    # required recipient fields). Anything else (incl. None / "inbound") uses the default prompt.
    kind: Optional[str] = PydField(default=None, max_length=20)
    user_id: Optional[str] = PydField(default=None, max_length=200)  # advisory only; budget is keyed by IP
    session_id: Optional[str] = PydField(default=None, max_length=200)  # stable id grouping turns of one conversation
    consent: bool = True  # may this turn be retained for model fine-tuning?

    @field_validator("schema")
    @classmethod
    def _cap_schema(cls, v: Optional[dict]) -> Optional[dict]:
        return _validate_schema_size(v)


class ChatResponse(BaseModel):
    schema: dict  # updated coltorapps schema, ready for builderStore / saveDraft
    title: str
    reply: str = ""  # natural-language message to show the user
    suggestions: List[str] = []  # clickable next-step ideas
    changed: bool = True  # False when the form was untouched (e.g. a question)
    action: Optional[str] = None  # lifecycle intent the app should run: checkin|publish|undo_checkout
    brain: str  # which LLM produced this ("groq" | "gemini" | "ollama")
    turn_id: Optional[str] = None  # transcript id; echo to /feedback to label this turn
    #: Web pages this turn was built from, when it did any research. Shown to the author so the
    #: form can be CHECKED: a menu scraped off the web can be out of date, and a wrong price
    #: becomes a real order at the wrong money. Empty for every ordinary turn.
    sources: List[dict] = []


class FeedbackRequest(BaseModel):
    """Attach a quality label to a recorded turn (by its turn_id), so the
    exporter can keep good examples and drop bad ones."""
    turn_id: str = PydField(max_length=64)
    accepted: Optional[bool] = None  # user kept (True) or undid (False) the edit
    rating: Optional[int] = PydField(default=None, ge=-1, le=1)  # +1 / -1 thumbs (bounded)
    note: Optional[str] = PydField(default=None, max_length=2000)


class TestDataRequest(BaseModel):
    """Ask LukeTests to generate test data to drive the builder's Test runs."""
    schema: Optional[dict] = None  # current coltorapps schema to generate values for
    mode: Literal["valid", "invalid"] = "valid"  # valid → should pass; invalid → should be rejected
    count: int = PydField(default=1, ge=1, le=5)  # distinct datasets (also clamped server-side)
    title: Optional[str] = PydField(default=None, max_length=500)
    user_id: Optional[str] = PydField(default=None, max_length=200)

    @field_validator("schema")
    @classmethod
    def _cap_schema(cls, v: Optional[dict]) -> Optional[dict]:
        return _validate_schema_size(v)


class TestDataResponse(BaseModel):
    datasets: List[TestDataItem]  # one or more {values, notes} datasets
    brain: str
