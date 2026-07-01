"""Intake normalizers + document chunking/aggregation for LukeSense.

These turn each intake type into plain text the classifier can judge, and roll a
long document's per-section judgements up into one overall judgement. All pure
functions — no LLM calls and no FastAPI — so they're trivially unit-testable; the
agent wires them to the shared brain.

Shapes are intentionally GENERIC (decoupled from any one repo): forms arrive as a
flat field-key -> value map (matching luke-capability-engine's FormInstance.data,
with labels resolved by the caller when available); emails as subject/body/sender;
documents as caller-extracted text.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, List, Optional

from .schema import SentimentAnalysis

# Per-field cap so one giant answer can't blow up tokens/cost.
_MAX_FIELD_CHARS = 1_000
_EXCERPT_CHARS = 160
# low < medium < high — used to take the MAX urgency across document sections.
_URGENCY_RANK = {"low": 0, "medium": 1, "high": 2}
# Tie-break order when two sentiments tie on summed confidence: surface problems.
_SENTIMENT_TIEBREAK = ["negative", "neutral", "positive"]

# Lines at/after these markers begin a quoted prior message — drop them (email).
_QUOTE_MARKERS = (
    re.compile(r"^\s*-{2,}\s*original message\s*-{2,}", re.I),
    re.compile(r"^\s*On .+ wrote:\s*$", re.I),
    re.compile(r"^\s*_{5,}\s*$"),            # Outlook divider
    re.compile(r"^\s*From:\s.+", re.I),      # forwarded header block
)
# Standard signature delimiter: a line that is exactly "-- ".
_SIG_DELIM = re.compile(r"^-- $")


def _render_value(val: Any) -> str:
    """Render one form answer value as compact text. Empty/blank -> ''."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        if not val:
            return ""
        if all(isinstance(x, dict) for x in val):  # data/edit grid rows
            rows = [
                "; ".join(f"{k}={v}" for k, v in row.items() if str(v).strip())
                for row in val
            ]
            return " | ".join(r for r in rows if r)
        return ", ".join(str(x).strip() for x in val if str(x).strip())
    if isinstance(val, dict):
        return ", ".join(f"{k}={v}" for k, v in val.items() if str(v).strip())
    return str(val).strip()


def normalize_form(
    data: dict, labels: Optional[dict] = None, title: Optional[str] = None
) -> str:
    """Render a form submission (FormInstance.data shape) as 'Label: value' lines.
    Skips empty answers; uses the field KEY when no label is supplied."""
    labels = labels or {}
    lines: List[str] = []
    if title:
        lines.append(f"Form: {title}")
    for key, raw in (data or {}).items():
        rendered = _render_value(raw)
        if not rendered:
            continue
        if len(rendered) > _MAX_FIELD_CHARS:
            rendered = rendered[:_MAX_FIELD_CHARS] + "…"
        lines.append(f"{labels.get(key, key)}: {rendered}")
    return "\n".join(lines)


def normalize_email(subject: Optional[str], body: str, sender: Optional[str] = None) -> str:
    """Strip quoted replies + signature from an email body, then prepend the
    sender/subject as context. Falls back to the raw body if stripping empties it."""
    kept: List[str] = []
    for line in (body or "").splitlines():
        if _SIG_DELIM.match(line):
            break  # signature block starts here
        if any(m.match(line) for m in _QUOTE_MARKERS):
            break  # quoted history starts here
        if line.lstrip().startswith(">"):
            continue  # inline quoted line
        kept.append(line)
    cleaned = "\n".join(kept).strip() or (body or "").strip()
    parts: List[str] = []
    if sender:
        parts.append(f"From: {sender}")
    if subject:
        parts.append(f"Subject: {subject}")
    if cleaned:
        parts.append(cleaned)
    return "\n".join(parts)


def chunk_text(text: str, max_chars: int) -> List[str]:
    """Split text into <= max_chars chunks on paragraph boundaries, hard-splitting
    any single paragraph longer than max_chars."""
    chunks: List[str] = []
    buf = ""
    for para in re.split(r"\n\s*\n", (text or "").strip()):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue
        if buf and len(buf) + len(para) + 2 > max_chars:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    if chunks:
        return chunks
    stripped = (text or "").strip()
    return [stripped] if stripped else []


def excerpt(text: str) -> str:
    """A short single-line preview of a section, for human triage."""
    one_line = " ".join((text or "").split())
    return one_line[:_EXCERPT_CHARS] + ("…" if len(one_line) > _EXCERPT_CHARS else "")


def aggregate(sections: List[SentimentAnalysis]) -> SentimentAnalysis:
    """Roll per-section judgements into one overall judgement. Urgency takes the
    MAX across sections (one urgent section makes the document urgent); sentiment
    is the class with the highest summed confidence (ties -> most negative);
    themes are unioned by frequency; the summary points at the most salient
    section. Deterministic — no extra LLM call."""
    if not sections:
        return SentimentAnalysis(summary="No sections to analyze.")
    urgency = max((s.urgency for s in sections), key=lambda u: _URGENCY_RANK[u])
    sums = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for s in sections:
        sums[s.sentiment] += max(s.confidence, 0.01)
    sentiment = max(
        _SENTIMENT_TIEBREAK,
        key=lambda c: (sums[c], -_SENTIMENT_TIEBREAK.index(c)),
    )
    total = sum(sums.values()) or 1.0
    confidence = round(sums[sentiment] / total, 2)
    themes = [t for t, _ in Counter(t for s in sections for t in s.themes).most_common(5)]
    salient = max(sections, key=lambda s: (_URGENCY_RANK[s.urgency], s.confidence))
    summary = f"{len(sections)} sections analyzed; most salient: {salient.summary}".strip()
    return SentimentAnalysis(
        sentiment=sentiment,
        confidence=confidence,
        urgency=urgency,
        themes=themes,
        summary=summary,
    )
