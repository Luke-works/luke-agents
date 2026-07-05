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
import getpass
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Iterable, Optional

log = logging.getLogger("luke_agents.export_finetune")


def _audit_export(**fields) -> None:
    """Record who/when/scope of an export (#29). Written to stderr and the app log, and
    appended to AGENTS_EXPORT_AUDIT_LOG (a JSONL path) when set, so exports of the training
    corpus leave an audit trail."""
    rec = {
        "event": "finetune_export",
        "at": datetime.now(timezone.utc).isoformat(),
        "actor": os.getenv("AGENTS_ACTOR") or _current_user(),
        **fields,
    }
    line = json.dumps(rec, ensure_ascii=False)
    print(f"audit: {line}", file=sys.stderr)
    log.info("finetune export audit: %s", line)
    path = os.getenv("AGENTS_EXPORT_AUDIT_LOG")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:  # audit must not silently vanish, but must not block the export
            print(f"audit: WARNING could not write {path}: {exc}", file=sys.stderr)


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "unknown"


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
def _read_postgres(
    dsn: str,
    schema: str,
    agent: Optional[str],
    tenant: Optional[str],
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: Optional[int] = None,
    itersize: int = 1000,
) -> Iterable[dict]:
    import psycopg2
    import psycopg2.extras

    clauses, params = [], []
    if agent:
        clauses.append("agent = %s")
        params.append(agent)
    if tenant:
        clauses.append("tenant_id = %s")  # tenant scoping (#33)
        params.append(tenant)
    if since:
        clauses.append("created_at >= %s")  # time-window bound (#38)
        params.append(since)
    if until:
        clauses.append("created_at <= %s")
        params.append(until)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT %s"
        params.append(limit)
    conn = psycopg2.connect(dsn)
    try:
        # NAMED (server-side) cursor so psycopg2 does NOT client-buffer the whole result
        # set; rows are streamed in `itersize` batches (#38). Unnamed cursors fetch it all.
        with conn.cursor(name="finetune_export", cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.itersize = itersize
            cur.execute(
                f"""SELECT messages, output, changed, accepted, rating, consent, error
                    FROM {schema}.turns {where} ORDER BY created_at{limit_sql}""",
                tuple(params),
            )
            for row in cur:
                yield dict(row)
    finally:
        conn.close()


def _read_jsonl(
    directory: str,
    agent: Optional[str],
    tenant: Optional[str],
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterable[dict]:
    import pathlib

    d = pathlib.Path(directory)
    turns_path, fb_path = d / "turns.jsonl", d / "feedback.jsonl"
    if not turns_path.exists():
        return

    # Join feedback (separate append-only file) onto turns by id; last write wins.
    # NOTE (#38): this feedback dict is bounded by the number of DISTINCT feedback'd
    # turns, not by corpus size, and JsonlStore is DEV-ONLY (Render's disk is ephemeral),
    # so we accept it. The turns file itself is streamed line-by-line below.
    feedback: dict[str, dict] = {}
    if fb_path.exists():
        with fb_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    fb = json.loads(line)
                    feedback.setdefault(fb["turn_id"], {}).update(
                        {k: v for k, v in fb.items() if k in ("accepted", "rating", "note") and v is not None}
                    )

    emitted = 0
    with turns_path.open(encoding="utf-8") as f:
        for line in f:  # stream — never load the whole turns file into memory (#38)
            if not line.strip():
                continue
            row = json.loads(line)
            if agent and row.get("agent") != agent:
                continue
            if tenant and row.get("tenant_id") != tenant:  # tenant scoping (#33)
                continue
            created = row.get("created_at")
            if since and created and created < since:
                continue
            if until and created and created > until:
                continue
            row.update(feedback.get(row.get("id"), {}))
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _rows(
    agent: Optional[str],
    tenant: Optional[str],
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterable[dict]:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        schema = os.getenv("AGENTS_DB_SCHEMA", "luke_agents")
        return _read_postgres(dsn, schema, agent, tenant, since=since, until=until, limit=limit)
    return _read_jsonl(
        os.getenv("TRANSCRIPTS_DIR", "data/transcripts"), agent, tenant,
        since=since, until=until, limit=limit,
    )


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Export recorded turns as fine-tuning JSONL.")
    p.add_argument("--agent", default=None, help="filter to one agent slug (e.g. 'form')")
    p.add_argument("--tenant", default=None, help="scope to one tenant id (required unless --all-tenants)")
    p.add_argument("--all-tenants", action="store_true",
                   help="export across ALL tenants (explicit opt-out of tenant scoping)")
    p.add_argument("--delete-tenant", default=None, metavar="TENANT",
                   help="ERASE every recorded turn for this tenant and exit (GDPR / off-boarding)")
    p.add_argument("--out", default="-", help="output .jsonl path, or '-' for stdout")
    p.add_argument("--only-accepted", action="store_true", help="require an explicit kept(=accepted) label")
    p.add_argument("--include-unchanged", action="store_true", help="also export chat/no-op turns")
    p.add_argument("--min-rating", type=int, default=None, help="require rating >= N")
    # Time-window / size bounds so a run streams a bounded slice (#38). --since/--until
    # compare against created_at (ISO-8601, e.g. 2026-01-01 or 2026-01-01T00:00:00+00:00).
    p.add_argument("--since", default=None, metavar="ISO", help="only turns with created_at >= this ISO timestamp")
    p.add_argument("--until", default=None, metavar="ISO", help="only turns with created_at <= this ISO timestamp")
    p.add_argument("--limit", type=int, default=None, help="stop after reading N turns (bounds a run)")
    args = p.parse_args(argv)

    if args.delete_tenant:
        from ..core.transcripts import delete_tenant
        n = delete_tenant(args.delete_tenant)
        _audit_export(action="delete_tenant", tenant=args.delete_tenant, erased=n)
        print(f"erased {n} turns for tenant {args.delete_tenant!r}", file=sys.stderr)
        return 0

    # Cross-tenant reads are prevented by default (#33): require an explicit tenant
    # scope, or an explicit --all-tenants opt-out.
    if not args.tenant and not args.all_tenants:
        p.error("specify --tenant <id> to scope the export, or --all-tenants to export across every tenant")

    # Audit the export request (who/when/scope) BEFORE streaming, so an interrupted or
    # OOMing run is still recorded (#29).
    _audit_export(
        action="export", agent=args.agent, tenant=args.tenant if not args.all_tenants else "*ALL*",
        since=args.since, until=args.until, limit=args.limit, out=args.out,
        only_accepted=args.only_accepted, include_unchanged=args.include_unchanged,
        min_rating=args.min_rating,
    )

    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    kept = total = 0
    try:
        for row in _rows(args.agent, args.tenant, since=args.since, until=args.until, limit=args.limit):
            total += 1
            if _keep(row, only_accepted=args.only_accepted,
                     include_unchanged=args.include_unchanged, min_rating=args.min_rating):
                out.write(json.dumps(_to_example(row), ensure_ascii=False) + "\n")
                kept += 1
    finally:
        if out is not sys.stdout:
            out.close()
    _audit_export(action="export_complete", kept=kept, total=total, out=args.out)
    print(f"exported {kept}/{total} turns -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
