"""initial turns table

The transcript store's base table, captured from the DDL that PostgresStore.init()
created (core/transcripts.py). The tenant_id column + index are a later revision
(0002), mirroring the real #33 evolution — a fresh DB ends at the same shape the app
expects. DDL is IF NOT EXISTS so applying this against a DB whose table was already
created by the app's init() fallback is a safe no-op.

Revision ID: 0001_initial_turns
Revises:
Create Date: 2026-07-25
"""
import os

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial_turns"
down_revision = None
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
        CREATE TABLE IF NOT EXISTS {s}.turns (
            id            uuid PRIMARY KEY,
            created_at    timestamptz NOT NULL DEFAULT now(),
            agent         text NOT NULL,
            brain         text,
            model         text,
            user_id       text,
            session_id    text,
            prompt_hash   text,
            messages      jsonb NOT NULL,
            output        jsonb,
            input_schema  jsonb,
            changed       boolean,
            latency_ms    integer,
            error         text,
            consent       boolean NOT NULL DEFAULT true,
            accepted      boolean,
            rating        smallint,
            feedback_note text,
            feedback_at   timestamptz
        );
    """)
    op.execute(
        f"CREATE INDEX IF NOT EXISTS turns_agent_created_idx ON {s}.turns (agent, created_at);"
    )


def downgrade() -> None:
    op.execute(f"DROP TABLE IF EXISTS {_schema()}.turns;")
