"""#31 — the transcript-schema migrations are ordered and well-formed.

A DB-free smoke test of the Alembic revision chain (the real apply/reverse/re-apply is
exercised against Postgres in the CI `migrations` job). Guards against a broken or
mis-linked revision landing silently.
"""
import importlib.util
import pathlib

import pytest

VERSIONS = pathlib.Path(__file__).resolve().parents[1] / "migrations" / "versions"


def _load(filename: str):
    pytest.importorskip("alembic")  # migration modules import `alembic.op` at module load
    path = VERSIONS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_revision_chain_is_ordered():
    r1 = _load("0001_initial_turns.py")
    r2 = _load("0002_add_tenant_id.py")
    r3 = _load("0003_audit_log.py")
    assert r1.revision == "0001_initial_turns"
    assert r1.down_revision is None                 # the base
    assert r2.down_revision == r1.revision          # links onto rev 1 (ordered)
    assert r3.down_revision == r2.revision          # rev 3 links onto rev 2
    # Both directions are defined so each migration is reversible.
    for mod in (r1, r2, r3):
        assert callable(mod.upgrade) and callable(mod.downgrade)


def test_schema_name_is_validated(monkeypatch):
    r1 = _load("0001_initial_turns.py")
    monkeypatch.setenv("AGENTS_DB_SCHEMA", "bad-schema; DROP TABLE x")
    with pytest.raises(ValueError):
        r1._schema()
