"""The form-combination matrix the agent's renderer is held to.

Every case here is a FormSpec the LLM could plausibly emit, rendered through the SAME
deterministic path the live agent uses (`spec_to_schema`). Two suites consume it:

  * `test_schema_key_safety.py` — asserts the structural invariants in Python.
  * `fixtures/agent-schema-cases.json` — the rendered schemas, checked into BOTH this repo and
    luke-forms, where `agentSchema.parity.test.ts` runs form-core's REAL `validateSchema` over
    them. That is the half Python cannot do: form-core owns the definition of a valid schema, so
    only form-core can confirm the agent produces one.

WHY IT EXISTS. A probe of 50 combinations on 2026-08-01 found SEVEN that rendered schemas
form-core rejects as errors — which block check-in and publish, i.e. the agent could build a form
its author could not ship. All were data keys: `Full Name`, `a.b`, `naïve`, `123`, an empty key,
two fields sharing a key, and a new field colliding with a container child the LLM never sees.

The type list is derived from the agent's own `FieldType`, so a newly-supported field type is
enrolled automatically rather than quietly untested.
"""
from __future__ import annotations

import typing
import uuid

from luke_agents.agents.form_agent.coltorapps import schema_to_spec, spec_to_schema
from luke_agents.agents.form_agent.schema import FieldType, FormSpec, SpecField, SpecLogicRule

#: Every field type the agent can emit — read off the LLM contract, never hand-listed.
ALL_TYPES: list[str] = list(typing.get_args(FieldType))

#: Every logic action the agent can emit.
ALL_ACTIONS = ["show", "hide", "enable", "disable", "require", "optional", "setValue"]

_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def _stabilise(schema: dict, seed: str) -> dict:
    """Replace generated entity ids with ids derived from the case name.

    The renderer mints uuid4s, so an unstabilised fixture would churn on every run and its diff
    would carry no information. uuid5 keeps them canonical UUIDs (which the builder requires)
    while making the file byte-stable, so a real change to the OUTPUT is the only thing that
    ever shows up in review.
    """
    ids = list(schema.get("entities", {}).keys())
    mapping = {old: str(uuid.uuid5(_NS, f"{seed}:{i}")) for i, old in enumerate(ids)}
    entities = {}
    for old, ent in schema["entities"].items():
        new_ent = dict(ent)
        if isinstance(ent.get("children"), list):
            new_ent["children"] = [mapping.get(c, c) for c in ent["children"]]
        entities[mapping[old]] = new_ent
    return {"entities": entities, "root": [mapping.get(r, r) for r in schema.get("root", [])]}


def _panel_schema() -> dict:
    """A container with a nested child — the shape the LLM never sees but shares a key namespace with."""
    return {
        "entities": {
            "11111111-1111-1111-1111-111111111111": {
                "type": "panel",
                "attributes": {"label": "Panel", "key": "panel"},
                "children": ["22222222-2222-2222-2222-222222222222"],
            },
            "22222222-2222-2222-2222-222222222222": {
                "type": "textField",
                "attributes": {"label": "Inner", "key": "inner"},
            },
        },
        "root": ["11111111-1111-1111-1111-111111111111"],
    }


def build_cases() -> list[dict]:
    """Every case as {name, schema}. Deterministic: same input, same bytes, every run."""
    cases: list[tuple[str, dict]] = []

    def case(name: str, spec: FormSpec, **kw) -> None:
        cases.append((name, spec_to_schema(spec, **kw)))

    # 1 ── each type on its own, with nothing set.
    for t in ALL_TYPES:
        case(f"bare:{t}", FormSpec(fields=[SpecField(key="f1", label="F", type=t)]))

    # 2 ── each type with EVERY optional attribute set at once. This is where an attribute that
    #      doesn't belong on a type (a placeholder on a checkbox, options on a button) surfaces.
    for t in ALL_TYPES:
        case(
            f"loaded:{t}",
            FormSpec(fields=[SpecField(
                key="f1", label="F", type=t, required=True, options=["a", "b"],
                placeholder="p", tooltip="t", description="d", hidden=True, disabled=True,
                calculate_value="1 + 1",
                logic=[SpecLogicRule(when='other == "x"', action="hide")],
            )]),
        )

    # 3 ── every logic action.
    for a in ALL_ACTIONS:
        case(
            f"logic:{a}",
            FormSpec(fields=[SpecField(
                key="f1", label="F",
                logic=[SpecLogicRule(when="a == 1", action=a, value="2" if a == "setValue" else None)],
            )]),
        )

    # 4 ── choice fields, which are the only ones that REQUIRE options.
    case("choice:no-options", FormSpec(fields=[SpecField(key="c", label="C", type="select")]))
    case("choice:empty-options", FormSpec(fields=[SpecField(key="c", label="C", type="select", options=[])]))
    case("choice:dup-options", FormSpec(fields=[SpecField(key="c", label="C", type="radio", options=["a", "a"])]))
    case("choice:blank-option", FormSpec(fields=[SpecField(key="c", label="C", type="selectBoxes", options=["", "b"])]))

    # 5 ── DATA KEYS. Every one of these rendered an unshippable form before 2026-08-01.
    case("key:duplicate", FormSpec(fields=[SpecField(key="dup", label="A"), SpecField(key="dup", label="B")]))
    case("key:triplicate", FormSpec(fields=[SpecField(key="d", label="A"), SpecField(key="d", label="B"), SpecField(key="d", label="C")]))
    case("key:empty", FormSpec(fields=[SpecField(key="", label="Empty")]))
    case("key:spaces", FormSpec(fields=[SpecField(key="has spaces", label="S")]))
    case("key:dots", FormSpec(fields=[SpecField(key="a.b", label="Dotted")]))
    case("key:dashes", FormSpec(fields=[SpecField(key="e-mail", label="Email")]))
    case("key:unicode", FormSpec(fields=[SpecField(key="naïve", label="U")]))
    case("key:cjk", FormSpec(fields=[SpecField(key="姓名", label="Name")]))
    case("key:numeric", FormSpec(fields=[SpecField(key="123", label="Age")]))
    case("key:leading-digit", FormSpec(fields=[SpecField(key="1st_choice", label="First choice")]))
    case("key:symbols-only", FormSpec(fields=[SpecField(key="!!!", label="???")]))
    case("key:reserved-true", FormSpec(fields=[SpecField(key="true", label="R")]))
    case("key:reserved-null", FormSpec(fields=[SpecField(key="null", label="R")]))
    case("key:reserved-undefined", FormSpec(fields=[SpecField(key="undefined", label="R")]))
    case("key:very-long", FormSpec(fields=[SpecField(key="k" * 300, label="Long")]))
    case("label:empty", FormSpec(fields=[SpecField(key="k", label="")]))
    case("label:and-key-empty", FormSpec(fields=[SpecField(key="", label="")]))

    # 6 ── a form using every type at once.
    case("mixed:all-types", FormSpec(fields=[
        SpecField(key=f"f_{t}", label=t, type=t, options=["x", "y"]) for t in ALL_TYPES
    ]))

    # 7 ── the MERGE path: editing a form that already exists.
    base = spec_to_schema(FormSpec(fields=[
        SpecField(key="k", label="L", type="textField", placeholder="p", required=True)
    ]))
    _, existing, pres, pres_root = schema_to_spec(base)
    merge_kw = {"existing": existing, "preserved_entities": pres, "preserved_root_ids": pres_root}
    case("merge:same-type", FormSpec(fields=[SpecField(key="k", label="L2", type="textField")]), **merge_kw)
    case("merge:type-change", FormSpec(fields=[SpecField(key="k", label="L2", type="select", options=["a"])]), **merge_kw)
    case("merge:rename-key", FormSpec(fields=[SpecField(key="k2", label="L", type="textField")]), **merge_kw)

    # 8 ── alongside a container whose children the LLM cannot see but shares a key namespace with.
    _, ex2, pres2, pr2 = schema_to_spec(_panel_schema())
    panel_kw = {"existing": ex2, "preserved_entities": pres2, "preserved_root_ids": pr2}
    case("preserved:container+new-field", FormSpec(fields=[SpecField(key="new", label="New")]), **panel_kw)
    case("preserved:key-collides-with-nested", FormSpec(fields=[SpecField(key="inner", label="Clash")]), **panel_kw)
    case("preserved:collides-after-normalising", FormSpec(fields=[SpecField(key="inner!", label="Clash")]), **panel_kw)
    case("preserved:empty-form", FormSpec(fields=[]), **panel_kw)

    # 9 ── nothing at all.
    case("empty:no-fields", FormSpec(fields=[]))

    return [{"name": n, "schema": _stabilise(s, n)} for n, s in cases]
