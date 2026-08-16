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

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_add_token_columns"
down_revision = "0003_audit_log"
branch_labels = None
depends_on = None

_IDENT = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _schema() -> str:
    s = os.getenv("AGENTS_DB_SCHEMA", "luke_agents")
    if not _IDENT.match(s):
        raise ValueError(f"invalid AGENTS_DB_SCHEMA: {s!r}")
    return s


# The schema name is interpolated into DDL because a SQL identifier CANNOT be a bound parameter,
# and it is validated against `_IDENT` above (and by the app before any migration runs), so it is
# not attacker-controllable. Same pattern as migrations 0002/0003. The `# nosemgrep` markers below
# silence the generic "formatted SQL" rule for these provably-safe identifier interpolations.
def upgrade() -> None:
    s = _schema()
    op.execute(f"ALTER TABLE {s}.turns ADD COLUMN IF NOT EXISTS prompt_tokens integer;")  # nosemgrep
    op.execute(f"ALTER TABLE {s}.turns ADD COLUMN IF NOT EXISTS completion_tokens integer;")  # nosemgrep


def downgrade() -> None:
    s = _schema()
    op.execute(f"ALTER TABLE {s}.turns DROP COLUMN IF EXISTS completion_tokens;")  # nosemgrep
    op.execute(f"ALTER TABLE {s}.turns DROP COLUMN IF EXISTS prompt_tokens;")  # nosemgrep
