# Data handling runbook — luke-agents transcripts

This service records conversation **turns** (the exact `messages` sent to the LLM plus the
model `output`) so they can later be exported as fine-tuning examples. That content can
contain PII — the form domain is literally about collecting names, emails, phone numbers,
addresses. This runbook documents how that data is classified, retained, minimized, and
erased. (GitHub issue #29.)

## What is stored

The `turns` table / `turns.jsonl` (see `luke_agents/core/transcripts.py`) holds, per turn:

| Field | Classification | Notes |
|-------|----------------|-------|
| `id`, `agent`, `brain`, `model`, `created_at`, `latency_ms`, `changed`, `error`, `prompt_hash` | Operational metadata | No user content. |
| `tenant_id`, `user_id`, `session_id` | Identifiers | `user_id`/`session_id` are client-supplied (advisory). Used for scoping + erasure. |
| `messages`, `output`, `input_schema` | **User content (may contain PII)** | The prompt (incl. the submitted form + user message) and the model's reply. |
| `consent` | Consent flag | Client-supplied per turn. Gates BOTH training export **and storage of content**. |
| `accepted`, `rating`, `feedback_note`, `feedback_at` | Quality labels | Attached later via `/feedback`. |

Backends: `PostgresStore` (prod, durable), `JsonlStore` (local dev only — Render's disk is
ephemeral), or `NullStore` (`TRANSCRIPTS_ENABLED=false`, record nothing).

## Consent gates storage (not just export)

When a request arrives with `consent=false`, the turn is stored **minimally**: metadata,
identifiers, timings, quality signals, and `error` are kept (needed for retention/erasure/
analytics), but `messages`/`output`/`input_schema` **content is dropped** before persistence
(`TurnRecord.for_storage()`). The `prompt_hash` is preserved so prompt-version grouping still
works. `consent=false` turns were already excluded from fine-tune export; now no user content
is retained for them at all.

## Redaction (optional)

Set `AGENTS_REDACT_PII=true` to best-effort scrub emails, phone numbers, and US SSNs from
stored content (`messages`/`output`/`input_schema`) before persistence. This is a
minimization aid, **not** a guarantee — it does not replace consent gating or retention, and
does not catch every PII form (names, free-text addresses). Off by default.

## Retention

Set `AGENTS_RETENTION_DAYS=N` to define the retention window. Nothing purges automatically in
the request path; enforce the window by running the purge command on a schedule (e.g. a daily
Render Cron Job or system cron):

```
python -m luke_agents.tools.retention purge            # uses AGENTS_RETENTION_DAYS
python -m luke_agents.tools.retention purge --days 90   # explicit override
```

Purge deletes turns whose `created_at` is older than the window (indexed by
`(agent, created_at)` and `(tenant_id, agent, created_at)`). Unset / `0` = keep forever.

## Right to erasure (GDPR / CCPA)

Erase all turns for a data subject or a whole tenant:

```
# One data subject (by user_id and/or session_id):
python -m luke_agents.tools.retention erase-user --user-id u-123
python -m luke_agents.tools.retention erase-user --session-id s-abc

# Entire tenant (off-boarding):
python -m luke_agents.tools.retention erase-tenant --tenant acme
```

These raise on failure (unlike the record path, which is fire-and-forget) so an operator sees
errors, and each action is audited to stderr and the app log.

## Export audit trail

Fine-tune exports (`luke_agents/tools/export_finetune.py`) and tenant erasures via that tool
write an audit record (event, timestamp, actor, scope, counts) to stderr and the app log, and
append it to the JSONL file at `AGENTS_EXPORT_AUDIT_LOG` when set. The actor is
`AGENTS_ACTOR` if set, else the OS user running the command. Exports can be bounded by
`--since` / `--until` (created_at ISO timestamps) and `--limit`, so a run is resumable by
time window and streams without loading the whole corpus.

## Quick reference — env vars

| Var | Effect |
|-----|--------|
| `AGENTS_RETENTION_DAYS` | Retention window in days (purge tool). Unset/0 = keep forever. |
| `AGENTS_REDACT_PII` | `true` → best-effort PII scrub before persistence. |
| `AGENTS_EXPORT_AUDIT_LOG` | JSONL path for export/erasure audit records. |
| `AGENTS_ACTOR` | Overrides the recorded actor for audit records. |
| `TRANSCRIPTS_ENABLED=false` | Record nothing. |
