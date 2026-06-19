"""Conversation transcript store — shared, agent-agnostic persistence of every
turn, so the data can later be exported as fine-tuning examples.

Each turn we record is already a supervised example: the exact `messages` sent
to the model (system + user) and the model's `output`. We also keep a quality
signal — the automatic `changed` flag plus optional user feedback (kept/undone,
👍/👎) attached later via the feedback endpoint — so the exporter can train on
GOOD turns only (see luke_agents/tools/export_finetune.py).

Backends, chosen at runtime:
  * TRANSCRIPTS_ENABLED=false        -> NullStore (record nothing).
  * DATABASE_URL set                 -> PostgresStore (durable; prod). Tables live
                                        in the AGENTS_DB_SCHEMA schema ("luke_agents").
  * otherwise                        -> JsonlStore (append-only files; local dev only,
                                        since Render's disk is ephemeral).

Recording must never break a user request: every public method swallows and logs
its own errors. Writes happen in a FastAPI BackgroundTask off the response path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("luke_agents.transcripts")

# Schema/identifier guard: we interpolate the schema name into DDL/SQL (you can't
# bind an identifier as a parameter), so it must be a plain SQL identifier.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnRecord:
    """One model turn = one (potential) fine-tuning example + its metadata."""
    id: str
    agent: str
    brain: str
    model: Optional[str]
    messages: list  # exact input: [{"role":"system",...}, {"role":"user",...}]
    output: Optional[dict]  # exact model output (the validated AssistantTurn dict)
    input_schema: Optional[dict] = None  # incoming coltorapps schema, for context
    tenant_id: Optional[str] = None  # owning tenant (#33) — set from the verified principal
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    changed: Optional[bool] = None  # auto quality signal: did the form change?
    latency_ms: Optional[int] = None
    error: Optional[str] = None  # set when the turn failed (excluded from training)
    consent: bool = True  # whether the caller allows training use
    created_at: str = field(default_factory=_now_iso)

    @property
    def prompt_hash(self) -> Optional[str]:
        """sha256 of the system prompt, so turns can be grouped by prompt version
        (system prompts evolve; you train on the one actually used)."""
        for m in self.messages:
            if m.get("role") == "system":
                return hashlib.sha256(m["content"].encode("utf-8")).hexdigest()
        return None


@dataclass
class Feedback:
    accepted: Optional[bool] = None  # user kept (True) or undid (False) the edit
    rating: Optional[int] = None  # +1 / -1 thumbs
    note: Optional[str] = None


# --------------------------------------------------------------------------- #
# Store implementations
# --------------------------------------------------------------------------- #
class TranscriptStore:
    name = "null"

    def init(self) -> None: ...
    def record_turn(self, rec: TurnRecord) -> None: ...
    def record_feedback(self, turn_id: str, fb: Feedback) -> bool:  # found?
        return False

    def delete_tenant(self, tenant_id: str) -> int:  # rows erased (#33)
        return 0


class NullStore(TranscriptStore):
    """Disabled: records nothing."""


class JsonlStore(TranscriptStore):
    """Append-only JSONL — for LOCAL DEV only. Turns and feedback go to separate
    files (append-only can't update a row); the exporter joins them by turn id."""
    name = "jsonl"

    def __init__(self, directory: str) -> None:
        self.dir = Path(directory)
        self.turns = self.dir / "turns.jsonl"
        self.feedback = self.dir / "feedback.jsonl"
        self._lock = threading.Lock()

    def init(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append(self, path: Path, obj: dict) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)  # lazy: don't rely on init()
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def record_turn(self, rec: TurnRecord) -> None:
        row = asdict(rec)
        row["prompt_hash"] = rec.prompt_hash
        self._append(self.turns, row)

    def record_feedback(self, turn_id: str, fb: Feedback) -> bool:
        self._append(self.feedback, {"turn_id": turn_id, "at": _now_iso(), **asdict(fb)})
        return True  # append-only can't confirm the turn exists; assume ok

    def delete_tenant(self, tenant_id: str) -> int:
        """Erase one tenant's turns (and their feedback) — rewrite both files,
        dropping matching rows. (#33; ties into retention.)"""
        with self._lock:
            if not self.turns.exists():
                return 0
            kept_turns, removed_ids = [], set()
            for line in self.turns.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("tenant_id") == tenant_id:
                    removed_ids.add(row.get("id"))
                else:
                    kept_turns.append(line)
            self.turns.write_text("\n".join(kept_turns) + ("\n" if kept_turns else ""), encoding="utf-8")
            if self.feedback.exists() and removed_ids:
                kept_fb = [
                    ln for ln in self.feedback.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and json.loads(ln).get("turn_id") not in removed_ids
                ]
                self.feedback.write_text("\n".join(kept_fb) + ("\n" if kept_fb else ""), encoding="utf-8")
            return len(removed_ids)


class PostgresStore(TranscriptStore):
    """Durable store. One table `<schema>.turns`; feedback updates the row in place."""
    name = "postgres"

    def __init__(self, dsn: str, schema: str) -> None:
        if not _IDENT.match(schema):
            raise ValueError(f"invalid schema name: {schema!r}")
        self.dsn = dsn
        self.schema = schema
        self._pool = None
        self._init_lock = threading.Lock()
        self._ready = False

    def _pool_or_connect(self):
        if self._pool is None:
            from psycopg2.pool import ThreadedConnectionPool
            self._pool = ThreadedConnectionPool(1, 5, self.dsn)
        return self._pool

    def init(self) -> None:
        # Idempotent: create schema + table if missing. Safe to call repeatedly.
        with self._init_lock:
            if self._ready:
                return
            ddl = f"""
            CREATE SCHEMA IF NOT EXISTS {self.schema};
            CREATE TABLE IF NOT EXISTS {self.schema}.turns (
                id            uuid PRIMARY KEY,
                created_at    timestamptz NOT NULL DEFAULT now(),
                agent         text NOT NULL,
                brain         text,
                model         text,
                tenant_id     text,
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
            CREATE INDEX IF NOT EXISTS turns_agent_created_idx
                ON {self.schema}.turns (agent, created_at);
            -- #33: migrate pre-existing tables, then index by tenant for scoped
            -- reads/exports and per-tenant erasure.
            ALTER TABLE {self.schema}.turns ADD COLUMN IF NOT EXISTS tenant_id text;
            CREATE INDEX IF NOT EXISTS turns_tenant_idx
                ON {self.schema}.turns (tenant_id, agent, created_at);
            """
            self._run(lambda cur: cur.execute(ddl))
            self._ready = True
            log.info("transcripts: postgres ready (schema=%s)", self.schema)

    def _run(self, fn):
        pool = self._pool_or_connect()
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    result = fn(cur)
            return result
        finally:
            pool.putconn(conn)

    def record_turn(self, rec: TurnRecord) -> None:
        from psycopg2.extras import Json

        if not self._ready:
            self.init()
        sql = f"""
            INSERT INTO {self.schema}.turns
              (id, agent, brain, model, tenant_id, user_id, session_id, prompt_hash,
               messages, output, input_schema, changed, latency_ms, error, consent)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO NOTHING
        """
        params = (
            rec.id, rec.agent, rec.brain, rec.model, rec.tenant_id, rec.user_id, rec.session_id,
            rec.prompt_hash, Json(rec.messages), Json(rec.output),
            Json(rec.input_schema), rec.changed, rec.latency_ms, rec.error, rec.consent,
        )
        self._run(lambda cur: cur.execute(sql, params))

    def record_feedback(self, turn_id: str, fb: Feedback) -> bool:
        if not self._ready:
            self.init()
        sql = f"""
            UPDATE {self.schema}.turns
               SET accepted = COALESCE(%s, accepted),
                   rating = COALESCE(%s, rating),
                   feedback_note = COALESCE(%s, feedback_note),
                   feedback_at = now()
             WHERE id = %s
        """

        def run(cur):
            cur.execute(sql, (fb.accepted, fb.rating, fb.note, turn_id))
            return cur.rowcount

        return bool(self._run(run))

    def delete_tenant(self, tenant_id: str) -> int:
        if not self._ready:
            self.init()
        sql = f"DELETE FROM {self.schema}.turns WHERE tenant_id = %s"

        def run(cur):
            cur.execute(sql, (tenant_id,))
            return cur.rowcount

        return int(self._run(run))


# --------------------------------------------------------------------------- #
# Selection + safe wrappers
# --------------------------------------------------------------------------- #
_store: Optional[TranscriptStore] = None
_store_lock = threading.Lock()


def _build_store() -> TranscriptStore:
    if os.getenv("TRANSCRIPTS_ENABLED", "true").strip().lower() in ("0", "false", "no"):
        return NullStore()
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        schema = os.getenv("AGENTS_DB_SCHEMA", "luke_agents")
        return PostgresStore(dsn, schema)
    return JsonlStore(os.getenv("TRANSCRIPTS_DIR", "data/transcripts"))


def get_store() -> TranscriptStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = _build_store()
    return _store


def safe_record_turn(rec: TurnRecord) -> None:
    """Persist a turn; never raise into the request/background path."""
    try:
        get_store().record_turn(rec)
    except Exception:  # noqa: BLE001
        log.exception("transcripts: failed to record turn %s", rec.id)


def safe_record_feedback(turn_id: str, fb: Feedback) -> bool:
    try:
        return get_store().record_feedback(turn_id, fb)
    except Exception:  # noqa: BLE001
        log.exception("transcripts: failed to record feedback for %s", turn_id)
        return False


def delete_tenant(tenant_id: str) -> int:
    """Erase every recorded turn for a tenant (GDPR / off-boarding, #33). Raises on
    failure — unlike the record paths, an erasure that silently failed would be a
    compliance hazard, so the caller (an ops tool) should see the error."""
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id is required for erasure")
    return get_store().delete_tenant(tenant_id.strip())
