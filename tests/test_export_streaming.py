"""#38 — fine-tune export streams and is bounded.

Verifies the JSONL reader streams line-by-line (does not read the whole file into
memory) and honours --since/--until/--limit, plus the export audit trail.
"""
import json

import luke_agents.tools.export_finetune as E


def _write_turns(tmp_path, rows):
    p = tmp_path / "turns.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _row(i, created, **kw):
    base = {
        "id": f"t{i}", "agent": "form", "created_at": created,
        "messages": [{"role": "user", "content": "hi"}], "output": {"reply": "ok"},
        "changed": True, "consent": True,
    }
    base.update(kw)
    return base


def test_read_jsonl_streams_line_by_line(tmp_path, monkeypatch):
    # If _read_jsonl slurps the whole file (read_text().splitlines()), patching
    # Path.read_text to blow up would break it. It must instead open() and iterate.
    _write_turns(tmp_path, [_row(i, f"2026-01-0{i}") for i in range(1, 4)])

    import pathlib
    orig_read_text = pathlib.Path.read_text

    def _boom(self, *a, **k):
        if self.name == "turns.jsonl":
            raise AssertionError("turns.jsonl must be streamed, not read_text()'d whole")
        return orig_read_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", _boom)
    rows = list(E._read_jsonl(str(tmp_path), agent=None, tenant=None))
    assert [r["id"] for r in rows] == ["t1", "t2", "t3"]


def test_read_jsonl_is_lazy_generator(tmp_path):
    # A huge file should not be materialised: pulling ONE item must not iterate all.
    _write_turns(tmp_path, [_row(i, f"2026-01-{i:02d}") for i in range(1, 32)])
    gen = E._read_jsonl(str(tmp_path), agent=None, tenant=None)
    first = next(iter(gen))
    assert first["id"] == "t1"  # produced without consuming the rest


def test_limit_bounds_the_run(tmp_path):
    _write_turns(tmp_path, [_row(i, f"2026-01-{i:02d}") for i in range(1, 11)])
    rows = list(E._read_jsonl(str(tmp_path), agent=None, tenant=None, limit=3))
    assert [r["id"] for r in rows] == ["t1", "t2", "t3"]


def test_since_until_window(tmp_path):
    _write_turns(tmp_path, [
        _row(1, "2026-01-01T00:00:00+00:00"),
        _row(2, "2026-02-15T00:00:00+00:00"),
        _row(3, "2026-03-20T00:00:00+00:00"),
    ])
    rows = list(E._read_jsonl(
        str(tmp_path), agent=None, tenant=None,
        since="2026-02-01T00:00:00+00:00", until="2026-03-01T00:00:00+00:00",
    ))
    assert [r["id"] for r in rows] == ["t2"]


def test_feedback_join_still_works_when_streaming(tmp_path):
    _write_turns(tmp_path, [_row(1, "2026-01-01")])
    (tmp_path / "feedback.jsonl").write_text(
        json.dumps({"turn_id": "t1", "rating": 1}) + "\n", encoding="utf-8"
    )
    rows = list(E._read_jsonl(str(tmp_path), agent=None, tenant=None))
    assert rows[0]["rating"] == 1


def test_main_exports_bounded_and_audits(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TRANSCRIPTS_DIR", str(tmp_path))
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AGENTS_EXPORT_AUDIT_LOG", str(audit_path))
    monkeypatch.setenv("AGENTS_ACTOR", "tester")
    _write_turns(tmp_path, [_row(i, f"2026-01-{i:02d}") for i in range(1, 11)])
    out = tmp_path / "out.jsonl"

    rc = E.main(["--all-tenants", "--limit", "2", "--out", str(out)])
    assert rc == 0
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 2  # limit honoured
    # audit trail written with actor + scope
    audit = [json.loads(l) for l in audit_path.read_text().splitlines() if l.strip()]
    events = {a["event"]: a for a in audit}
    assert "finetune_export" in events
    assert any(a.get("action") == "export" and a.get("actor") == "tester" for a in audit)
    assert any(a.get("action") == "export_complete" for a in audit)
