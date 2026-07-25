"""add tenant_id column + index

Captures the #33 per-tenant isolation change as an ordered, reviewable migration —
the exact column the app's init() adds inline today. Demonstrates evolving the table
after it exists (AC: "adding a column is demonstrated via a migration"). IF NOT EXISTS
makes it safe against a DB where init() already added the column.

Revision ID: 0002_add_tenant_id
Revises: 0001_initial_turns
Create Date: 2026-07-25
"""
import os

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_add_tenant_id"
down_revision = "0001_initial_turns"
branch_labels = None
depends_on = None

_IDENT = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _schema() -> str:
    s = os.getenv("AGENTS_DB_SCHEMA", "luke_agents")
    if not _IDENT.match(s):
        raise ValueError(f"invalid AGENTS_DB_SCHEMA: {s!r}")
    return s


def upgrade() -> None:
    s = _schema()
    op.execute(f"ALTER TABLE {s}.turns ADD COLUMN IF NOT EXISTS tenant_id text;")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS turns_tenant_idx ON {s}.turns (tenant_id, agent, created_at);"
    )


def downgrade() -> None:
    s = _schema()
    op.execute(f"DROP INDEX IF EXISTS {s}.turns_tenant_idx;")
    op.execute(f"ALTER TABLE {s}.turns DROP COLUMN IF EXISTS tenant_id;")
