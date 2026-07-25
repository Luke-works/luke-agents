"""append-only audit_log for sensitive actions

Records who exported the training corpus and who changed a turn's quality label
(#40), with a verified actor, action, scope, request id, and timestamp. Append-only
by convention (no UPDATE/DELETE except retention purge). IF NOT EXISTS so it is a
safe no-op against a DB where the app's init() already created it.

Revision ID: 0003_audit_log
Revises: 0002_add_tenant_id
Create Date: 2026-07-25
"""
import os

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_audit_log"
down_revision = "0002_add_tenant_id"
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
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {s}.audit_log (
            id          uuid PRIMARY KEY,
            at          timestamptz NOT NULL DEFAULT now(),
            actor       text NOT NULL,
            action      text NOT NULL,
            target      text,
            scope       text,
            request_id  text,
            details     jsonb
        );
    """)
    op.execute(f"CREATE INDEX IF NOT EXISTS audit_log_at_idx ON {s}.audit_log (at);")
    op.execute(f"CREATE INDEX IF NOT EXISTS audit_log_action_idx ON {s}.audit_log (action, at);")


def downgrade() -> None:
    op.execute(f"DROP TABLE IF EXISTS {_schema()}.audit_log;")
