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
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("luke_agents.transcripts")

# Schema/identifier guard: we interpolate the schema name into DDL/SQL (you can't
# bind an identifier as a parameter), so it must be a plain SQL identifier.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _is_prod() -> bool:
    """Production deployment marker — reuses the AGENTS_ENV convention the server's
    assert_prod_hardened() already keys on."""
    return os.getenv("AGENTS_ENV", "").strip().lower() in ("prod", "production")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Retention / PII configuration (#29)
# --------------------------------------------------------------------------- #
def retention_days() -> Optional[int]:
    """How long to keep turns, in days. 0 / unset / non-positive = keep forever
    (no automatic purge). Configured via AGENTS_RETENTION_DAYS."""
    raw = os.getenv("AGENTS_RETENTION_DAYS", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        log.warning("AGENTS_RETENTION_DAYS=%r is not an integer; ignoring", raw)
        return None
    return n if n > 0 else None


def _redaction_enabled() -> bool:
    return os.getenv("AGENTS_REDACT_PII", "").strip().lower() in ("1", "true", "yes")


# Conservative PII patterns. Redaction is OPT-IN (AGENTS_REDACT_PII=true) and best-effort:
# it scrubs the most common direct identifiers from stored content so the training corpus
# minimizes retained PII. It is NOT a substitute for consent gating or retention.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d(?:[\s\-().]{0,2}\d){6,}(?!\w)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")


def _redact_text(s: str) -> str:
    s = _EMAIL_RE.sub("[redacted-email]", s)
    s = _SSN_RE.sub("[redacted-ssn]", s)
    s = _PHONE_RE.sub("[redacted-phone]", s)
    return s


def redact(value):
    """Recursively redact common PII from stored strings (best-effort). Applied to
    `messages`, `output`, and `input_schema` before persistence when AGENTS_REDACT_PII
    is on. Structure is preserved; only string leaves are scrubbed."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


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
    # #64: durable per-turn token usage (from llm.last_usage()) — auditable per-tenant/period
    # history to bill and report against, independent of the live Prometheus counter.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    consent: bool = True  # whether the caller allows training use
    created_at: str = field(default_factory=_now_iso)
    # Set by for_storage() to preserve the real prompt hash when content is dropped.
    _prompt_hash_override: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def prompt_hash(self) -> Optional[str]:
        """sha256 of the system prompt, so turns can be grouped by prompt version
        (system prompts evolve; you train on the one actually used)."""
        if self._prompt_hash_override is not None:
            return self._prompt_hash_override
        for m in self.messages:
            content = m.get("content")
            if m.get("role") == "system" and content:
                return hashlib.sha256(content.encode("utf-8")).hexdigest()
        return None

    def for_storage(self) -> "TurnRecord":
        """Return a copy ready to persist, applying consent + redaction (#29).

        - consent=False → store MINIMALLY: keep metadata (id, tenant, timings, error,
          quality signals) for retention/erasure/analytics, but DROP the message and
          output content so no user-submitted text is retained without consent.
        - AGENTS_REDACT_PII=true → best-effort scrub PII from retained content.
        The prompt_hash is computed from the ORIGINAL system prompt before any dropping,
        so grouping-by-prompt-version still works on consent=False turns.
        """
        # Preserve prompt_hash of the real system prompt before we possibly drop content.
        original_hash = self.prompt_hash
        messages, output, input_schema = self.messages, self.output, self.input_schema
        if self.consent is False:
            # Minimal record: no user/system content, no model output, no incoming schema.
            messages = [
                {"role": m.get("role"), "content": ""} for m in self.messages
            ]
            output = None
            input_schema = None
        elif _redaction_enabled():
            messages = redact(self.messages)
            output = redact(self.output)
            input_schema = redact(self.input_schema)
        rec = TurnRecord(
            id=self.id, agent=self.agent, brain=self.brain, model=self.model,
            messages=messages, output=output, input_schema=input_schema,
            tenant_id=self.tenant_id, user_id=self.user_id, session_id=self.session_id,
            changed=self.changed, latency_ms=self.latency_ms, error=self.error,
            prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
            consent=self.consent, created_at=self.created_at,
        )
        rec._prompt_hash_override = original_hash
        return rec


@dataclass
class Feedback:
    accepted: Optional[bool] = None  # user kept (True) or undid (False) the edit
    rating: Optional[int] = None  # +1 / -1 thumbs
    note: Optional[str] = None


@dataclass
class AuditRecord:
    """One append-only audit entry for a sensitive action (#40): a training-corpus export
    or a turn label change. The `actor` is a VERIFIED principal (the gateway-set tenant, or
    an authenticated operator) — never a client-supplied field."""
    id: str
    actor: str
    action: str  # e.g. "feedback.label", "finetune.export", "tenant.delete"
    target: Optional[str] = None       # the thing acted on (turn id, tenant, output path)
    scope: Optional[str] = None        # tenant / filter scope
    request_id: Optional[str] = None   # request correlation id, or a CLI run id
    details: Optional[dict] = None
    created_at: str = field(default_factory=_now_iso)


# --------------------------------------------------------------------------- #
# Store implementations
# --------------------------------------------------------------------------- #
class TranscriptStore:
    name = "null"
    ephemeral = False  # True only for stores whose data doesn't survive a redeploy (#28)

    def init(self) -> None: ...
    def record_turn(self, rec: TurnRecord) -> None: ...
    def record_feedback(self, turn_id: str, fb: Feedback) -> bool:  # found?
        return False

    def record_audit(self, rec: "AuditRecord") -> None:  # append-only audit (#40)
        ...

    def list_audit(self, *, action: Optional[str] = None, limit: int = 100) -> list:  # queryable (#40)
        return []

    def ping(self) -> bool:  # readiness check (#35) — is the store reachable?
        return True

    def delete_tenant(self, tenant_id: str) -> int:  # rows erased (#33)
        return 0

    def delete_for_user(self, *, user_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> int:  # right-to-erasure (#29)
        return 0

    def purge_older_than(self, days: int) -> int:  # retention (#29)
        return 0


class NullStore(TranscriptStore):
    """Disabled: records nothing."""


class JsonlStore(TranscriptStore):
    """Append-only JSONL — for LOCAL DEV only. Turns and feedback go to separate
    files (append-only can't update a row); the exporter joins them by turn id."""
    name = "jsonl"
    ephemeral = True  # Render's disk is ephemeral — never a prod backend (#28)

    def __init__(self, directory: str) -> None:
        self.dir = Path(directory)
        self.turns = self.dir / "turns.jsonl"
        self.feedback = self.dir / "feedback.jsonl"
        self.audit = self.dir / "audit.jsonl"
        self._lock = threading.Lock()

    def init(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append(self, path: Path, obj: dict) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)  # lazy: don't rely on init()
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def record_turn(self, rec: TurnRecord) -> None:
        rec = rec.for_storage()  # apply consent gating + optional redaction (#29)
        row = asdict(rec)
        row.pop("_prompt_hash_override", None)  # internal-only, don't persist
        row["prompt_hash"] = rec.prompt_hash
        self._append(self.turns, row)

    def record_feedback(self, turn_id: str, fb: Feedback) -> bool:
        self._append(self.feedback, {"turn_id": turn_id, "at": _now_iso(), **asdict(fb)})
        return True  # append-only can't confirm the turn exists; assume ok

    def record_audit(self, rec: "AuditRecord") -> None:
        self._append(self.audit, asdict(rec))

    def list_audit(self, *, action: Optional[str] = None, limit: int = 100) -> list:
        if not self.audit.exists():
            return []
        rows = [json.loads(ln) for ln in self.audit.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if action:
            rows = [r for r in rows if r.get("action") == action]
        return rows[-limit:]

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
            self._drop_feedback_for(removed_ids)
            return len(removed_ids)

    def _drop_feedback_for(self, removed_ids: set) -> None:
        """Rewrite feedback.jsonl dropping rows whose turn_id was erased. Caller holds the lock."""
        if self.feedback.exists() and removed_ids:
            kept_fb = [
                ln for ln in self.feedback.read_text(encoding="utf-8").splitlines()
                if ln.strip() and json.loads(ln).get("turn_id") not in removed_ids
            ]
            self.feedback.write_text("\n".join(kept_fb) + ("\n" if kept_fb else ""), encoding="utf-8")

    def _rewrite_turns(self, keep) -> int:
        """Rewrite turns.jsonl keeping only rows where keep(row) is True; drop matching
        turns' feedback too. Returns the number of turns removed. Caller must NOT hold the lock."""
        with self._lock:
            if not self.turns.exists():
                return 0
            kept_lines, removed_ids = [], set()
            for line in self.turns.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if keep(row):
                    kept_lines.append(line)
                else:
                    removed_ids.add(row.get("id"))
            self.turns.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
            self._drop_feedback_for(removed_ids)
            return len(removed_ids)

    def delete_for_user(self, *, user_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> int:
        """Right-to-erasure (#29): drop every turn matching the given user_id and/or
        session_id (a row matches if ALL supplied identifiers match)."""
        if not user_id and not session_id:
            raise ValueError("delete_for_user requires user_id and/or session_id")

        def keep(row: dict) -> bool:
            if user_id is not None and row.get("user_id") != user_id:
                return True
            if session_id is not None and row.get("session_id") != session_id:
                return True
            return False  # all supplied ids matched → erase

        return self._rewrite_turns(keep)

    def purge_older_than(self, days: int) -> int:
        """Retention (#29): drop turns whose created_at is older than `days` days."""
        if days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        def keep(row: dict) -> bool:
            created = row.get("created_at")
            if not created:
                return True  # no timestamp → keep (can't judge age)
            try:
                ts = datetime.fromisoformat(created)
            except ValueError:
                return True
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts >= cutoff

        return self._rewrite_turns(keep)


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
            # #30: size the pool via env (relative to uvicorn workers/threadpool). Defaults
            # match the historical 1..5. Exhaustion raises PoolError, which _write_with_retry
            # treats as transient and retries with backoff rather than dropping the turn.
            mn = max(1, _int_env("AGENTS_DB_POOL_MIN", 1))
            mx = max(mn, _int_env("AGENTS_DB_POOL_MAX", 5))
            self._pool = ThreadedConnectionPool(mn, mx, self.dsn)
        return self._pool

    def init(self) -> None:
        # Idempotent runtime safety-net: create schema + table if missing. The AUTHORITATIVE,
        # versioned schema lives in Alembic migrations (migrations/, run as a pre-deploy step,
        # #31); this keeps a Postgres-backed dev box or a first boot working before/without a
        # separate `alembic upgrade head`. Both use IF NOT EXISTS and converge on the same shape.
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
                prompt_tokens     integer,
                completion_tokens integer,
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
            -- #64: durable per-turn token history. Authoritative DDL lives in migration 0004;
            -- this keeps a Postgres-backed dev box / first boot working before it runs.
            ALTER TABLE {self.schema}.turns ADD COLUMN IF NOT EXISTS prompt_tokens integer;
            ALTER TABLE {self.schema}.turns ADD COLUMN IF NOT EXISTS completion_tokens integer;
            CREATE INDEX IF NOT EXISTS turns_tenant_idx
                ON {self.schema}.turns (tenant_id, agent, created_at);
            -- #40: append-only audit of sensitive actions (export, label change). Authoritative
            -- DDL lives in migration 0003; this keeps a first boot working before it runs.
            CREATE TABLE IF NOT EXISTS {self.schema}.audit_log (
                id            uuid PRIMARY KEY,
                at            timestamptz NOT NULL DEFAULT now(),
                actor         text NOT NULL,
                action        text NOT NULL,
                target        text,
                scope         text,
                request_id    text,
                details       jsonb
            );
            CREATE INDEX IF NOT EXISTS audit_log_at_idx ON {self.schema}.audit_log (at);
            CREATE INDEX IF NOT EXISTS audit_log_action_idx ON {self.schema}.audit_log (action, at);
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
        rec = rec.for_storage()  # apply consent gating + optional redaction (#29)
        sql = f"""
            INSERT INTO {self.schema}.turns
              (id, agent, brain, model, tenant_id, user_id, session_id, prompt_hash,
               messages, output, input_schema, changed, latency_ms, error,
               prompt_tokens, completion_tokens, consent)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO NOTHING
        """
        params = (
            rec.id, rec.agent, rec.brain, rec.model, rec.tenant_id, rec.user_id, rec.session_id,
            rec.prompt_hash, Json(rec.messages), Json(rec.output),
            Json(rec.input_schema), rec.changed, rec.latency_ms, rec.error,
            rec.prompt_tokens, rec.completion_tokens, rec.consent,
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

    def ping(self) -> bool:
        # Readiness (#35): a real round-trip to Postgres. Never raises — a False here means
        # "not ready", handled by the caller (readiness probe returns 503).
        try:
            self._run(lambda cur: cur.execute("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False

    def record_audit(self, rec: "AuditRecord") -> None:
        from psycopg2.extras import Json

        if not self._ready:
            self.init()
        sql = f"""
            INSERT INTO {self.schema}.audit_log
              (id, actor, action, target, scope, request_id, details)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO NOTHING
        """
        params = (rec.id, rec.actor, rec.action, rec.target, rec.scope, rec.request_id, Json(rec.details))
        self._run(lambda cur: cur.execute(sql, params))

    def list_audit(self, *, action: Optional[str] = None, limit: int = 100) -> list:
        if not self._ready:
            self.init()
        where, params = "", []
        if action:
            where = "WHERE action = %s"
            params.append(action)
        params.append(max(1, limit))
        sql = (
            f"SELECT id, at, actor, action, target, scope, request_id, details "
            f"FROM {self.schema}.audit_log {where} ORDER BY at DESC LIMIT %s"
        )

        def run(cur):
            cur.execute(sql, tuple(params))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

        return self._run(run)

    def delete_tenant(self, tenant_id: str) -> int:
        if not self._ready:
            self.init()
        sql = f"DELETE FROM {self.schema}.turns WHERE tenant_id = %s"

        def run(cur):
            cur.execute(sql, (tenant_id,))
            return cur.rowcount

        return int(self._run(run))

    def delete_for_user(self, *, user_id: Optional[str] = None,
                        session_id: Optional[str] = None) -> int:
        """Right-to-erasure (#29): DELETE turns matching the given user_id and/or
        session_id (ALL supplied identifiers must match)."""
        if not self._ready:
            self.init()
        if not user_id and not session_id:
            raise ValueError("delete_for_user requires user_id and/or session_id")
        clauses, params = [], []
        if user_id is not None:
            clauses.append("user_id = %s")
            params.append(user_id)
        if session_id is not None:
            clauses.append("session_id = %s")
            params.append(session_id)
        sql = f"DELETE FROM {self.schema}.turns WHERE {' AND '.join(clauses)}"

        def run(cur):
            cur.execute(sql, tuple(params))
            return cur.rowcount

        return int(self._run(run))

    def purge_older_than(self, days: int) -> int:
        """Retention (#29): DELETE turns older than `days` days by created_at."""
        if not self._ready:
            self.init()
        if days <= 0:
            return 0
        sql = (
            f"DELETE FROM {self.schema}.turns "
            f"WHERE created_at < now() - make_interval(days => %s)"
        )

        def run(cur):
            cur.execute(sql, (days,))
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
    # No DATABASE_URL. JsonlStore is a DEV-ONLY backend (Render's disk is ephemeral).
    # #28: in production, refuse to silently downgrade durable storage to ephemeral files —
    # a config slip would lose the training corpus on the next redeploy with no signal. Warn
    # loudly and record NOTHING (NullStore) rather than pretend to persist.
    if _is_prod():
        log.warning(
            "transcripts: ENABLED but DATABASE_URL is unset in production (AGENTS_ENV=%s) — "
            "refusing ephemeral JSONL (data would vanish on redeploy); recording is DISABLED. "
            "Set DATABASE_URL to persist transcripts.",
            os.getenv("AGENTS_ENV", ""),
        )
        return NullStore()
    return JsonlStore(os.getenv("TRANSCRIPTS_DIR", "data/transcripts"))


def get_store() -> TranscriptStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = _build_store()
    return _store


# --------------------------------------------------------------------------- #
# Durability: metrics, bounded retry, and an off-path write queue (#41, #30)
# --------------------------------------------------------------------------- #
_metrics_lock = threading.Lock()
_metrics = {"turns_written": 0, "turns_dropped": 0, "write_retries": 0, "feedback_dropped": 0}


def _incr(name: str, n: int = 1) -> None:
    with _metrics_lock:
        _metrics[name] = _metrics.get(name, 0) + n


def metrics() -> dict:
    """Snapshot of transcript-write counters (surfaced in /health). Makes silent write loss
    observable; ties into the metrics/Prometheus work (#22)."""
    with _metrics_lock:
        return dict(_metrics)


# psycopg2 transient failures (connection dropped, pool exhausted) are worth retrying; a
# programming/constraint error is not. Matched by class name so we needn't import psycopg2
# here (it's absent in dev/JSONL mode).
_RETRYABLE_EXC = {"PoolError", "OperationalError", "InterfaceError"}


def _is_transient(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        if type(cur).__name__ in _RETRYABLE_EXC:
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


def _with_retry(fn, *, what: str):
    """Run fn(), retrying transient DB/pool errors with exponential backoff. Re-raises the
    last error if it's non-transient or attempts are exhausted."""
    attempts = max(1, _int_env("AGENTS_TRANSCRIPT_WRITE_RETRIES", 3))
    base = _float_env("AGENTS_TRANSCRIPT_RETRY_BASE_SECONDS", 0.1)
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not _is_transient(exc) or i == attempts - 1:
                raise
            _incr("write_retries")
            log.warning("transcripts: transient %s on %s (attempt %d/%d): %s",
                        type(exc).__name__, what, i + 1, attempts, exc)
            time.sleep(base * (2 ** i))


class _TurnWriter:
    """Single-threaded, bounded, off-path queue for durable turn writes (#41). Decouples the
    write from the request lifecycle, retries transient failures, counts drops, and can be
    flushed on graceful shutdown."""

    def __init__(self, maxsize: int) -> None:
        self._q: "queue.Queue[Optional[TurnRecord]]" = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is None:
                t = threading.Thread(target=self._loop, name="transcript-writer", daemon=True)
                t.start()
                self._thread = t

    def submit(self, rec: TurnRecord) -> None:
        self._ensure_started()
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            _incr("turns_dropped")
            log.warning("transcripts: write queue full (max=%s) — dropped turn %s",
                        self._q.maxsize, rec.id)

    def _loop(self) -> None:
        while True:
            rec = self._q.get()
            try:
                if rec is None:  # shutdown sentinel (unused; the thread is a daemon)
                    return
                try:
                    _with_retry(lambda: get_store().record_turn(rec), what="record_turn")
                    _incr("turns_written")
                except Exception as exc:  # noqa: BLE001
                    _incr("turns_dropped")
                    log.warning("transcripts: dropped turn %s after retries (%s: %s)",
                                rec.id, type(exc).__name__, exc)
            finally:
                self._q.task_done()

    def flush(self, timeout: float) -> int:
        """Best-effort: wait up to `timeout`s for queued writes to drain. Returns the number
        still pending (0 if fully flushed)."""
        deadline = time.monotonic() + max(0.0, timeout)
        while self._q.unfinished_tasks > 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        return self._q.unfinished_tasks


_writer: Optional[_TurnWriter] = None
_writer_lock = threading.Lock()


def _get_writer() -> _TurnWriter:
    global _writer
    if _writer is None:
        with _writer_lock:
            if _writer is None:
                _writer = _TurnWriter(_int_env("AGENTS_TRANSCRIPT_QUEUE_MAX", 1000))
    return _writer


def safe_record_turn(rec: TurnRecord) -> None:
    """Enqueue a turn for durable, retried, off-request-path persistence (#41). Returns
    immediately and never raises into the caller; the write (with backoff) runs on the writer
    thread and is drained on graceful shutdown via flush_pending()."""
    try:
        _get_writer().submit(rec)
    except Exception:  # noqa: BLE001
        _incr("turns_dropped")
        log.exception("transcripts: failed to enqueue turn %s", rec.id)


def flush_pending(timeout: float = 5.0) -> int:
    """Give queued transcript writes a chance to complete on graceful shutdown (#41). No-op
    if nothing was ever enqueued. Returns the number still pending after `timeout`."""
    if _writer is None:
        return 0
    pending = _get_writer().flush(timeout)
    if pending:
        log.warning("transcripts: %d queued turn(s) still pending after %.1fs flush", pending, timeout)
    return pending


def safe_record_feedback(turn_id: str, fb: Feedback) -> bool:
    """Record feedback synchronously (the endpoint returns whether the turn was found),
    retrying transient failures. Never raises; a final failure is counted + logged."""
    try:
        return bool(_with_retry(lambda: get_store().record_feedback(turn_id, fb), what="record_feedback"))
    except Exception:  # noqa: BLE001
        _incr("feedback_dropped")
        log.warning("transcripts: failed to record feedback for %s after retries", turn_id)
        return False


def safe_record_audit(rec: AuditRecord) -> None:
    """Write an append-only audit entry (#40), retrying transient failures. Never raises into
    the caller — an audit-store hiccup must not break the feedback request or the export run
    (the export also keeps its stderr/file trail)."""
    try:
        _with_retry(lambda: get_store().record_audit(rec), what="record_audit")
    except Exception:  # noqa: BLE001
        log.warning("transcripts: failed to write audit '%s' (%s) after retries", rec.action, rec.id)


def list_audit(*, action: Optional[str] = None, limit: int = 100) -> list:
    """Query recent audit entries (newest first). Ops/compliance read path (#40)."""
    return get_store().list_audit(action=action, limit=limit)


def delete_tenant(tenant_id: str) -> int:
    """Erase every recorded turn for a tenant (GDPR / off-boarding, #33). Raises on
    failure — unlike the record paths, an erasure that silently failed would be a
    compliance hazard, so the caller (an ops tool) should see the error."""
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id is required for erasure")
    return get_store().delete_tenant(tenant_id.strip())


def delete_for_user(user_id: Optional[str] = None, session_id: Optional[str] = None) -> int:
    """Right-to-erasure for one data subject (#29): erase all turns for a user_id
    and/or session_id. Raises on failure (an ops tool should see it)."""
    user_id = (user_id or "").strip() or None
    session_id = (session_id or "").strip() or None
    if not user_id and not session_id:
        raise ValueError("a user_id and/or session_id is required for erasure")
    return get_store().delete_for_user(user_id=user_id, session_id=session_id)


def purge_older_than(days: Optional[int] = None) -> int:
    """Purge turns older than `days` (defaults to AGENTS_RETENTION_DAYS). Returns the
    number of rows purged; 0 (and no-op) when retention is disabled. Raises on failure."""
    if days is None:
        days = retention_days()
    if not days or days <= 0:
        return 0
    return get_store().purge_older_than(days)
