"""add per-turn token columns (prompt_tokens / completion_tokens)

Captures the #64 durable token-history change as an ordered, reviewable migration — the
exact columns the app's init() adds inline today. Gives auditable per-tenant/period token
usage to bill and report against, independent of the live Prometheus counter. IF NOT EXISTS
makes it safe against a DB where init() already added the columns.

Revision ID: 0004_add_token_columns
Revises: 0003_audit_log
Create Date: 2026-08-16
"""
import os

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_add_token_columns"
down_revision = "0003_audit_log"
branch_labels = None
depends_on = None

_IDENT = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_COLUMNS = ("prompt_tokens", "completion_tokens")


def _schema() -> str:
    s = os.getenv("AGENTS_DB_SCHEMA", "luke_agents")
    if not _IDENT.match(s):
        raise ValueError(f"invalid AGENTS_DB_SCHEMA: {s!r}")
    return s


def _existing_columns(schema: str) -> set:
    """Columns already on <schema>.turns — makes this migration idempotent against a DB where the
    app's inline init() DDL already added them (mirrors the IF-NOT-EXISTS intent of 0002/0003)."""
    insp = sa.inspect(op.get_bind())
    return {c["name"] for c in insp.get_columns("turns", schema=schema)}


# Uses Alembic's structured DDL API (op.add_column / op.drop_column) rather than a formatted SQL
# string — no identifier interpolation, and idempotent via the existence check above.
def upgrade() -> None:
    s = _schema()
    have = _existing_columns(s)
    for col in _COLUMNS:
        if col not in have:
            op.add_column("turns", sa.Column(col, sa.Integer(), nullable=True), schema=s)


def downgrade() -> None:
    s = _schema()
    have = _existing_columns(s)
    for col in reversed(_COLUMNS):
        if col in have:
            op.drop_column("turns", col, schema=s)
