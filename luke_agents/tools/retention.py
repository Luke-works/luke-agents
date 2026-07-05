"""Data-retention & right-to-erasure management CLI (#29).

Operates on the same transcript store the app uses (Postgres in prod, JSONL in dev).
Intended to be run on a schedule (e.g. a daily cron / Render cron job) for purge, and
on demand for erasure (GDPR/CCPA data-subject requests).

Usage:
    # Purge turns older than the retention window (AGENTS_RETENTION_DAYS), or --days N:
    python -m luke_agents.tools.retention purge
    python -m luke_agents.tools.retention purge --days 90

    # Right-to-erasure: erase all turns for a data subject (by user_id and/or session_id):
    python -m luke_agents.tools.retention erase-user --user-id u-123
    python -m luke_agents.tools.retention erase-user --session-id s-abc

    # Erase an entire tenant (off-boarding):
    python -m luke_agents.tools.retention erase-tenant --tenant acme

Exit code is 0 on success. Actions are audited to stderr (and the app log).
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from ..core.transcripts import (
    delete_for_user,
    delete_tenant,
    purge_older_than,
    retention_days,
)

log = logging.getLogger("luke_agents.retention")


def _audit(action: str, detail: str) -> None:
    line = f"retention: {action} — {detail}"
    print(line, file=sys.stderr)
    log.info(line)


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Transcript retention & erasure management (#29).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("purge", help="delete turns older than the retention window")
    pp.add_argument("--days", type=int, default=None,
                    help="override retention window (default: AGENTS_RETENTION_DAYS)")

    eu = sub.add_parser("erase-user", help="right-to-erasure for one data subject")
    eu.add_argument("--user-id", default=None)
    eu.add_argument("--session-id", default=None)

    et = sub.add_parser("erase-tenant", help="erase every turn for a tenant (off-boarding)")
    et.add_argument("--tenant", required=True)

    args = p.parse_args(argv)

    if args.cmd == "purge":
        days = args.days if args.days is not None else retention_days()
        if not days or days <= 0:
            _audit("purge", "no retention window configured (set AGENTS_RETENTION_DAYS or --days); nothing purged")
            return 0
        n = purge_older_than(days)
        _audit("purge", f"deleted {n} turns older than {days} days")
        return 0

    if args.cmd == "erase-user":
        if not args.user_id and not args.session_id:
            p.error("erase-user requires --user-id and/or --session-id")
        n = delete_for_user(user_id=args.user_id, session_id=args.session_id)
        _audit("erase-user", f"deleted {n} turns (user_id={args.user_id!r}, session_id={args.session_id!r})")
        return 0

    if args.cmd == "erase-tenant":
        n = delete_tenant(args.tenant)
        _audit("erase-tenant", f"deleted {n} turns for tenant {args.tenant!r}")
        return 0

    p.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
