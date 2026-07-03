"""EmailDoc schema — the email agent's LLM contract.

The LLM emits a small, BOUNDED `EmailDoc` block document (see the Email Template
Builder contract §1). The same shape is rendered by luke-consumer-ui and stored
(as JSON) by luke-core-engine, so it must stay identical across all three repos.

The block list is a discriminated union on the `type` field — only the seven
block kinds below are allowed; anything else is forbidden (the prompt enforces
this, and the agent repairs/drops anything that slips through). Variables of the
form `{{identifier}}` inside any string are PRESERVED literally — they are the
Postmark merge contract, not something the model should fill in.
"""
from __future__ import annotations

import json
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field as PydField, field_validator
from typing_extensions import Annotated

# Bound request size so a single call can't amplify cost or exhaust memory
# (these endpoints are unauthenticated and publicly reachable).
_MAX_MESSAGE_CHARS = 8_000
_MAX_DOC_BYTES = 200_000
_MAX_BLOCKS = 50

# theme.contentWidth is clamped to this inclusive px range (email-safe layout).
_MIN_CONTENT_WIDTH = 480
_MAX_CONTENT_WIDTH = 700
_DEFAULT_CONTENT_WIDTH = 600

# Theme defaults — also the server-side repair fallbacks.
_DEFAULT_BRAND_COLOR = "#2563eb"
_DEFAULT_BACKGROUND_COLOR = "#f3f4f6"
_DEFAULT_CONTENT_BACKGROUND = "#ffffff"

Align = Literal["left", "center", "right"]
FontFamily = Literal["sans", "serif", "mono"]
HeadingLevel = Literal[1, 2, 3]


def _validate_doc_size(v: Optional[dict]) -> Optional[dict]:
    if v is not None and len(json.dumps(v, default=str)) > _MAX_DOC_BYTES:
        raise ValueError("doc is too large")
    return v


# The enum-ish fields below are COERCED to an allowed value (not rejected) when the
# model emits a near-miss — this is the contract's "coerce level/align to allowed
# values" repair, applied at parse time so a slightly-off field never 502s the turn.
def _coerce_align(v: object) -> str:
    return v if v in ("left", "center", "right") else "left"


def _coerce_level(v: object) -> int:
    try:
        n = int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return n if n in (1, 2, 3) else 1


def _coerce_font(v: object) -> str:
    return v if v in ("sans", "serif", "mono") else "sans"


class Theme(BaseModel):
    """Visual theme for the whole email. `contentWidth` is clamped 480..700."""
    brandColor: str = PydField(
        default=_DEFAULT_BRAND_COLOR, description="Hex accent colour used for buttons/links."
    )
    fontFamily: FontFamily = PydField(default="sans", description="'sans' | 'serif' | 'mono'.")
    contentWidth: int = PydField(
        default=_DEFAULT_CONTENT_WIDTH, description="Card width in px, between 480 and 700."
    )
    backgroundColor: str = PydField(
        default=_DEFAULT_BACKGROUND_COLOR, description="Hex page background colour."
    )
    contentBackground: str = PydField(
        default=_DEFAULT_CONTENT_BACKGROUND, description="Hex card (content) background colour."
    )

    @field_validator("contentWidth")
    @classmethod
    def _clamp_width(cls, v: int) -> int:
        return max(_MIN_CONTENT_WIDTH, min(int(v), _MAX_CONTENT_WIDTH))

    @field_validator("fontFamily", mode="before")
    @classmethod
    def _coerce_font(cls, v: object) -> str:
        return _coerce_font(v)


# --------------------------------------------------------------------------- #
# Block union — discriminated on `type`. These are the ONLY allowed block kinds.
# --------------------------------------------------------------------------- #
class HeadingBlock(BaseModel):
    type: Literal["heading"] = "heading"
    text: str = PydField(description="Heading text. May contain {{vars}} (keep them literal).")
    level: HeadingLevel = PydField(default=1, description="1, 2 or 3 (default 1).")
    align: Align = PydField(default="left")

    @field_validator("level", mode="before")
    @classmethod
    def _coerce_level(cls, v: object) -> int:
        return _coerce_level(v)

    @field_validator("align", mode="before")
    @classmethod
    def _coerce_align(cls, v: object) -> str:
        return _coerce_align(v)


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str = PydField(
        description="Paragraph text. Markdown-lite: **bold**, *italic*, [label](url). "
        "Keep any {{vars}} literal."
    )
    align: Align = PydField(default="left")

    @field_validator("align", mode="before")
    @classmethod
    def _coerce_align(cls, v: object) -> str:
        return _coerce_align(v)


class ButtonBlock(BaseModel):
    type: Literal["button"] = "button"
    label: str = PydField(description="Button caption.")
    href: str = PydField(description="Destination URL (https) or a {{var}}.")
    align: Align = PydField(default="left")
    bgColor: Optional[str] = PydField(
        default=None, description="Hex background; defaults to theme.brandColor when null."
    )
    textColor: Optional[str] = PydField(
        default=None, description="Hex text colour; defaults to white when null."
    )

    @field_validator("align", mode="before")
    @classmethod
    def _coerce_align(cls, v: object) -> str:
        return _coerce_align(v)


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    src: str = PydField(description="Image URL (https) or a {{var}}.")
    alt: str = PydField(description="Alt text for accessibility.")
    width: Optional[int] = PydField(default=None, description="Display width in px.")
    href: Optional[str] = PydField(default=None, description="Optional link the image points to.")
    align: Align = PydField(default="center")

    @field_validator("align", mode="before")
    @classmethod
    def _coerce_align(cls, v: object) -> str:
        return _coerce_align(v)


# The exact set of allowed block `type` values — used to drop unknown blocks
# before the discriminated union validates them (server-side repair).
_BLOCK_TYPES = {"heading", "text", "button", "image", "divider", "spacer", "footer"}


class DividerBlock(BaseModel):
    type: Literal["divider"] = "divider"


class SpacerBlock(BaseModel):
    type: Literal["spacer"] = "spacer"
    size: int = PydField(default=24, description="Vertical space in px (default 24).")


class FooterBlock(BaseModel):
    type: Literal["footer"] = "footer"
    text: str = PydField(description="Footer text, e.g. address · {{companyName}}.")
    unsubscribeUrl: Optional[str] = PydField(
        default=None, description="Unsubscribe link; usually the {{unsubscribeUrl}} var."
    )


# Discriminated on `type`: Pydantic routes each block to the one matching model
# (clean errors, no "tried every member" noise) and rejects any other `type`.
Block = Annotated[
    Union[
        HeadingBlock,
        TextBlock,
        ButtonBlock,
        ImageBlock,
        DividerBlock,
        SpacerBlock,
        FooterBlock,
    ],
    PydField(discriminator="type"),
]


class EmailDoc(BaseModel):
    """The full email document the LLM returns each turn (max 50 blocks)."""
    subject: str = PydField(default="", description="Email subject. May contain {{vars}}.")
    preheader: Optional[str] = PydField(
        default=None, description="Optional inbox preview text shown after the subject."
    )
    theme: Theme = PydField(default_factory=Theme)
    blocks: List[Block] = PydField(
        default_factory=list,
        description="Ordered list of blocks (max 50). Discriminated on `type`.",
    )

    @field_validator("blocks", mode="before")
    @classmethod
    def _drop_unknown_blocks(cls, v: object) -> object:
        """Server-side repair: silently drop entries whose `type` is not in the
        bounded vocabulary (or aren't objects), so one stray block from the model
        never 502s the whole turn. Known blocks then validate via the union."""
        if not isinstance(v, list):
            return v
        return [
            b for b in v
            if isinstance(b, BaseModel)
            or (isinstance(b, dict) and b.get("type") in _BLOCK_TYPES)
        ]

    @field_validator("blocks")
    @classmethod
    def _cap_blocks(cls, v: List[Block]) -> List[Block]:
        if len(v) > _MAX_BLOCKS:
            raise ValueError(f"too many blocks (max {_MAX_BLOCKS})")
        return v


class ChatRequest(BaseModel):
    """One turn. Stateless: the client (the Email Builder) sends the current
    EmailDoc back each time, so the agent needs no storage."""
    message: str = PydField(max_length=_MAX_MESSAGE_CHARS)
    # Current EmailDoc: {"subject":..., "theme":{...}, "blocks":[...]}. Optional /
    # null for a brand-new email.
    doc: Optional[dict] = None
    title: Optional[str] = PydField(default=None, max_length=500)
    user_id: Optional[str] = PydField(default=None, max_length=200)  # advisory only; budget is keyed by IP
    session_id: Optional[str] = PydField(default=None, max_length=200)  # stable id grouping turns of one conversation
    consent: bool = True  # may this turn be retained for model fine-tuning?

    @field_validator("doc")
    @classmethod
    def _cap_doc(cls, v: Optional[dict]) -> Optional[dict]:
        return _validate_doc_size(v)


class ChatResponse(BaseModel):
    doc: dict  # full updated EmailDoc, ready for the renderer / saveDraft
    title: str
    reply: str = ""  # natural-language message to show the user
    suggestions: List[str] = []  # clickable next-step ideas
    changed: bool = True  # False when the email was untouched (e.g. a question)
    brain: str  # which LLM produced this ("groq" | "openai" | "gemini" | "ollama")
    turn_id: Optional[str] = None  # transcript id


class TestDataItem(BaseModel):
    """One generated set of sample variable values."""
    values: dict = PydField(
        default_factory=dict,
        description="Map of variable name -> a plausible sample value, e.g. "
        "{'firstName': 'Jordan', 'companyName': 'Acme'}.",
    )


class TestDataTurn(BaseModel):
    """What the LLM returns when asked to generate sample variable values."""
    samples: List[TestDataItem] = PydField(
        default_factory=list,
        description="COUNT distinct sample sets — meaningfully different realistic personas.",
    )


class TestDataRequest(BaseModel):
    """Ask the agent to invent plausible values for each {{var}} in the doc
    (for preview + test send)."""
    doc: Optional[dict] = None  # current EmailDoc to pull variables from
    count: int = PydField(default=1, ge=1, le=5)  # distinct sample sets (also clamped server-side)
    user_id: Optional[str] = PydField(default=None, max_length=200)

    @field_validator("doc")
    @classmethod
    def _cap_doc(cls, v: Optional[dict]) -> Optional[dict]:
        return _validate_doc_size(v)


class TestDataResponse(BaseModel):
    samples: List[TestDataItem]  # one or more {values} sample sets
    brain: str
