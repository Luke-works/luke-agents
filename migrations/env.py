"""Alembic environment for the luke-agents transcript schema (#31).

The DB URL comes from DATABASE_URL and the target schema from AGENTS_DB_SCHEMA
(default 'luke_agents') — the exact env the app's PostgresStore reads — so migrations
apply to precisely the store the app writes to. Alembic's own version table lives in
that schema, and the schema is created first if missing.
"""
from __future__ import annotations

import os
import re
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool, text

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Same identifier guard the app uses — the schema is interpolated into DDL.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _schema() -> str:
    schema = os.getenv("AGENTS_DB_SCHEMA", "luke_agents")
    if not _IDENT.match(schema):
        raise ValueError(f"invalid AGENTS_DB_SCHEMA: {schema!r}")
    return schema


def _url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is unset — nothing to migrate. Transcripts fall back to dev "
            "JSONL without it; migrations only apply to the Postgres backend."
        )
    # Render/Heroku hand out postgres://; SQLAlchemy requires the postgresql:// scheme.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=None,
        literal_binds=True,
        version_table_schema=_schema(),
        include_schemas=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    schema = _schema()
    engine = create_engine(_url(), poolclass=pool.NullPool, future=True)
    with engine.connect() as connection:
        # Ensure the target schema exists before the version table is created in it.
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=None,
            version_table_schema=schema,
            include_schemas=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
