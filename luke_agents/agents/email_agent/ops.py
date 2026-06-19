"""Server-side sanitize/repair for the EmailDoc the LLM returns.

This is the email agent's equivalent of the contract's `repairEmailDoc` (mirrored
in luke-consumer-ui's `emailDoc.ts`): before we hand a doc back to the UI we drop
unknown block types, clamp `theme.contentWidth`, fill theme defaults, and coerce
`level`/`align` to allowed values — so the UI ALWAYS gets a valid, bounded doc no
matter what the model emitted.

It also extracts the merge-variable set ({{var}} names) and derives the small
conversational bits (`reply`, `suggestions`) the chat response needs but that the
EmailDoc response_model doesn't carry — the LLM returns the doc only.
"""
from __future__ import annotations

import re
from typing import List

from .schema import (
    _DEFAULT_CONTENT_WIDTH,
    _MAX_BLOCKS,
    _MAX_CONTENT_WIDTH,
    _MIN_CONTENT_WIDTH,
    EmailDoc,
    Theme,
)

# A "variable" is any {{identifier}} (mirrors the contract regex exactly).
_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

# Where variables may appear, per block type (the string-bearing fields).
_VAR_FIELDS = ("text", "label", "href", "src", "unsubscribeUrl")


def repair_doc(doc: EmailDoc) -> EmailDoc:
    """Clamp/fill an EmailDoc so the UI always gets a valid, bounded document.

    The Pydantic model + its validators already drop unknown block types (the
    discriminated union rejects them), clamp `contentWidth`, and coerce
    `level`/`align` to the allowed literals at validation time. This pass makes
    the remaining guarantees explicit and idempotent: theme defaults are filled
    and the block list is hard-capped at the bound.
    """
    theme = doc.theme or Theme()
    # Re-run through Theme() so any missing fields fall back to defaults and
    # contentWidth is re-clamped (idempotent — safe if already valid).
    width = theme.contentWidth or _DEFAULT_CONTENT_WIDTH
    theme = Theme(
        brandColor=theme.brandColor,
        fontFamily=theme.fontFamily,
        contentWidth=max(_MIN_CONTENT_WIDTH, min(int(width), _MAX_CONTENT_WIDTH)),
        backgroundColor=theme.backgroundColor,
        contentBackground=theme.contentBackground,
    )
    blocks = list(doc.blocks[:_MAX_BLOCKS])  # hard cap (defensive; validator also bounds it)
    return EmailDoc(
        subject=doc.subject or "",
        preheader=doc.preheader,
        theme=theme,
        blocks=blocks,
    )


def extract_variables(doc: EmailDoc) -> List[str]:
    """Distinct {{var}} names across subject + every string-bearing block field,
    in first-seen order (the merge contract shown in the UI / sent to Postmark)."""
    seen: dict[str, None] = {}

    def scan(text: str | None) -> None:
        if not text:
            return
        for m in _VAR_RE.finditer(text):
            seen.setdefault(m.group(1), None)

    scan(doc.subject)
    scan(doc.preheader)
    for block in doc.blocks:
        for field in _VAR_FIELDS:
            scan(getattr(block, field, None))
    return list(seen.keys())


def derive_reply(changed: bool, doc: EmailDoc) -> str:
    """A short, friendly natural-language reply for the chat response. The LLM
    returns the doc only (response_model is EmailDoc), so we phrase the reply
    deterministically here."""
    if not changed:
        return "Got it — I left your email as-is. Tell me what you'd like to change."
    n = len(doc.blocks)
    count = "1 block" if n == 1 else f"{n} blocks"
    return f"Done — your email now has {count}. Want me to tweak anything else?"


def derive_suggestions(doc: EmailDoc) -> List[str]:
    """A few contextual next-step ideas, based on what the doc is missing."""
    types = {b.type for b in doc.blocks}
    out: List[str] = []
    if "footer" not in types:
        out.append("Add a footer with unsubscribe")
    if "button" not in types:
        out.append("Add a call-to-action button")
    if "image" not in types:
        out.append("Add your logo image")
    if not extract_variables(doc):
        out.append("Personalize the greeting")
    out.append("Adjust the brand color")
    return out[:4]
