"""Export recorded turns as fine-tuning JSONL.

Reads the transcript store (Postgres or JSONL — same selection as the app) and
writes one JSON object per line in the chat format every major provider accepts:

    {"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}

The assistant message is the model's exact JSON output (what we want it to learn
to produce). Only GOOD turns are exported — see the filter below — so you don't
train the model on its own mistakes.

Usage:
    python -m luke_agents.tools.export_finetune --agent form --out form_sft.jsonl

    # stricter: only turns the user explicitly kept, drop 👎
    python -m luke_agents.tools.export_finetune --only-accepted --out form_sft.jsonl

Selection (default): error IS NULL, output present, consent=true, rating != -1,
and the turn changed the form (changed=true) OR was explicitly accepted. Add
--include-unchanged to also export chat/no-op turns, --only-accepted to require
an explicit keep, --min-rating N to require a rating.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, Optional


def _keep(row: dict, *, only_accepted: bool, include_unchanged: bool, min_rating: Optional[int]) -> bool:
    if row.get("error") or not row.get("output"):
        return False
    if row.get("consent") is False:
        return False
    rating = row.get("rating")
    if rating is not None and rating < 0:
        return False  # explicit 👎
    if min_rating is not None and (rating is None or rating < min_rating):
        return False
    accepted = row.get("accepted")
    if accepted is False:
        return False  # user undid the edit
    if only_accepted:
        return accepted is True
    # An explicit positive label is the strongest signal — keep it regardless of
    # the changed-gate below.
    if accepted is True or (rating is not None and rating > 0):
        return True
    if not include_unchanged and not row.get("changed"):
        return False  # skip pure chat / no-op turns for a form-edit fine-tune
    return True


def _to_example(row: dict) -> dict:
    messages = list(row["messages"])
    messages.append({"role": "assistant", "content": json.dumps(row["output"], ensure_ascii=False)})
    return {"messages": messages}


# --------------------------------------------------------------------------- #
# Readers — mirror the app's store selection
# --------------------------------------------------------------------------- #
def _read_postgres(dsn: str, schema: str, agent: Optional[str]) -> Iterable[dict]:
    import psycopg2
    import psycopg2.extras

    where = "WHERE agent = %s" if agent else ""
    params = (agent,) if agent else ()
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""SELECT messages, output, changed, accepted, rating, consent, error
                    FROM {schema}.turns {where} ORDER BY created_at""",
                params,
            )
            for row in cur:
                yield dict(row)
    finally:
        conn.close()


def _read_jsonl(directory: str, agent: Optional[str]) -> Iterable[dict]:
    import pathlib

    d = pathlib.Path(directory)
    turns_path, fb_path = d / "turns.jsonl", d / "feedback.jsonl"
    if not turns_path.exists():
        return

    # Join feedback (separate append-only file) onto turns by id; last write wins.
    feedback: dict[str, dict] = {}
    if fb_path.exists():
        for line in fb_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                fb = json.loads(line)
                feedback.setdefault(fb["turn_id"], {}).update(
                    {k: v for k, v in fb.items() if k in ("accepted", "rating", "note") and v is not None}
                )

    for line in turns_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if agent and row.get("agent") != agent:
            continue
        row.update(feedback.get(row.get("id"), {}))
        yield row


def _rows(agent: Optional[str]) -> Iterable[dict]:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        schema = os.getenv("AGENTS_DB_SCHEMA", "luke_agents")
        return _read_postgres(dsn, schema, agent)
    return _read_jsonl(os.getenv("TRANSCRIPTS_DIR", "data/transcripts"), agent)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Export recorded turns as fine-tuning JSONL.")
    p.add_argument("--agent", default=None, help="filter to one agent slug (e.g. 'form')")
    p.add_argument("--out", default="-", help="output .jsonl path, or '-' for stdout")
    p.add_argument("--only-accepted", action="store_true", help="require an explicit kept(=accepted) label")
    p.add_argument("--include-unchanged", action="store_true", help="also export chat/no-op turns")
    p.add_argument("--min-rating", type=int, default=None, help="require rating >= N")
    args = p.parse_args(argv)

    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    kept = total = 0
    try:
        for row in _rows(args.agent):
            total += 1
            if _keep(row, only_accepted=args.only_accepted,
                     include_unchanged=args.include_unchanged, min_rating=args.min_rating):
                out.write(json.dumps(_to_example(row), ensure_ascii=False) + "\n")
                kept += 1
    finally:
        if out is not sys.stdout:
            out.close()
    print(f"exported {kept}/{total} turns -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
