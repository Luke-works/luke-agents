"""The renderer must never emit a form its author cannot ship.

form-core rejects an invalid or duplicated data key as an ERROR, and errors block check-in and
publish. So a schema this agent renders with a bad key is not a cosmetic problem: it is an
AI-built form that stops dead at the Problems panel.

These assertions mirror `@lukeflow/form-core`'s own rules. The other half of the contract —
running form-core's REAL validator over the same cases — lives in luke-forms and is fed by
`fixtures/agent-schema-cases.json`; see `form_matrix.py`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from luke_agents.agents.form_agent.coltorapps import KNOWN_FIELD_TYPES, spec_to_schema
from luke_agents.agents.form_agent.schema import FormSpec, SpecField

from tests.form_matrix import ALL_TYPES, build_cases

# Mirrors form-core KEY_REGEX_SOURCE / RESERVED_KEYS. If form-core tightens these, the parity
# fixture in luke-forms fails first — that is the point of shipping the rendered cases there.
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RESERVED_KEYS = {"true", "false", "null", "undefined", "NaN", "Infinity"}

CASES = build_cases()
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "agent-schema-cases.json"


def _keys(schema: dict) -> list[str]:
    return [
        (e.get("attributes") or {}).get("key")
        for e in schema["entities"].values()
        if isinstance(e, dict)
    ]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_every_key_is_a_valid_identifier(case):
    for key in _keys(case["schema"]):
        assert isinstance(key, str) and key, f"{case['name']}: empty/missing key"
        assert KEY_RE.match(key), f"{case['name']}: {key!r} is not a valid identifier"
        assert key not in RESERVED_KEYS, f"{case['name']}: {key!r} is reserved"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_keys_are_unique_across_the_whole_form(case):
    # Across EVERY entity, not just the ones the LLM authored: a container's nested children
    # share the submission namespace, and two fields with one key means colliding answers.
    keys = _keys(case["schema"])
    assert len(keys) == len(set(keys)), f"{case['name']}: duplicate keys {keys}"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_root_and_children_reference_real_entities(case):
    schema = case["schema"]
    ids = set(schema["entities"])
    for rid in schema["root"]:
        assert rid in ids, f"{case['name']}: root references missing entity {rid}"
    for eid, ent in schema["entities"].items():
        for child in ent.get("children") or []:
            assert child in ids, f"{case['name']}: {eid} references missing child {child}"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_entity_ids_are_canonical_uuids(case):
    # The builder validates entity ids as canonical UUIDs; a truncated hex string is rejected.
    uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
    for eid in case["schema"]["entities"]:
        assert uuid_re.match(eid), f"{case['name']}: {eid!r} is not a canonical UUID"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_choice_fields_always_have_options(case):
    for ent in case["schema"]["entities"].values():
        if ent.get("type") in {"select", "radio", "selectBoxes"}:
            opts = (ent.get("attributes") or {}).get("options")
            assert isinstance(opts, list) and opts, f"{case['name']}: choice field without options"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_button_carries_no_value_attributes(case):
    # `button` has no value, so required/placeholder/options/calculateValue are not part of its
    # attribute set — emitting one produces an "Unknown entity attribute" schema.
    for ent in case["schema"]["entities"].values():
        if ent.get("type") == "button":
            attrs = ent.get("attributes") or {}
            for banned in ("required", "placeholder", "options", "calculateValue", "tooltip", "description"):
                assert banned not in attrs, f"{case['name']}: button carries {banned}"


def test_the_matrix_covers_every_supported_field_type():
    """A newly-supported type must not slip in untested."""
    covered = {t for t in ALL_TYPES for c in CASES if c["name"] == f"bare:{t}"}
    assert covered == set(ALL_TYPES)
    # And the LLM contract must not drift from what the renderer knows how to build.
    assert set(ALL_TYPES) == set(KNOWN_FIELD_TYPES), (
        "FieldType (the LLM contract) and KNOWN_FIELD_TYPES (the renderer) disagree"
    )


# ── specific regressions, named so a failure says what broke ──────────────────

def test_a_label_rescues_an_unusable_key():
    schema = spec_to_schema(FormSpec(fields=[SpecField(key="123", label="Age")]))
    assert _keys(schema) == ["age"], "a numeric key should fall back to the label, not to 'field'"


def test_accents_decompose_rather_than_vanish():
    schema = spec_to_schema(FormSpec(fields=[SpecField(key="naïve", label="N")]))
    assert _keys(schema) == ["naive"]


def test_duplicate_keys_are_suffixed_in_order():
    schema = spec_to_schema(FormSpec(fields=[
        SpecField(key="dup", label="A"), SpecField(key="dup", label="B"), SpecField(key="dup", label="C"),
    ]))
    assert _keys(schema) == ["dup", "dup_2", "dup_3"], "the first field should keep the plain key"


def test_an_existing_nested_child_keeps_its_key_and_the_new_field_yields():
    from tests.form_matrix import _panel_schema
    from luke_agents.agents.form_agent.coltorapps import schema_to_spec

    _, existing, pres, pres_root = schema_to_spec(_panel_schema())
    schema = spec_to_schema(
        FormSpec(fields=[SpecField(key="inner", label="Clash")]),
        existing=existing, preserved_entities=pres, preserved_root_ids=pres_root,
    )
    keys = _keys(schema)
    assert "inner" in keys and "inner_2" in keys, keys
    # The PRESERVED child must be the one that kept `inner` — renaming it would silently move
    # data that already exists under that key.
    inner_owner = [e for e in schema["entities"].values() if (e.get("attributes") or {}).get("key") == "inner"]
    assert inner_owner[0]["attributes"]["label"] == "Inner"


def test_a_valid_author_chosen_key_is_left_alone():
    # Only synthesised keys get house style; camelCase an author picked must survive untouched.
    schema = spec_to_schema(FormSpec(fields=[SpecField(key="firstName", label="First name")]))
    assert _keys(schema) == ["firstName"]


# ── the cross-language fixture ────────────────────────────────────────────────

def test_fixture_matches_the_matrix():
    """The committed fixture is what luke-forms validates; it must be regenerated when the
    renderer changes, or that repo is checking a schema this one no longer produces."""
    assert FIXTURE.exists(), f"missing {FIXTURE} — run `python -m tests.regen_fixture`"
    on_disk = json.loads(FIXTURE.read_text())
    assert on_disk["cases"] == CASES, (
        "fixtures/agent-schema-cases.json is stale — regenerate it with "
        "`python -m tests.regen_fixture` and copy it into luke-forms/fixtures/"
    )
