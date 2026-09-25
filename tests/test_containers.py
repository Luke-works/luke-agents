"""Layout containers: the agent can build tabs, panels and wizard steps, and edit inside them.

From the reported session: the user asked for "Tabs for each section" and a stepper per dish and
was told "I can't add Tabs or Stepper components here. The builder only has basic field types,
with no layout containers like tabs or panels and no dedicated stepper control." The builder had
all three. The agent's spec was a FLAT list of 13 types, so a container was not merely missing
from a list — it was unrepresentable.

These pin the four things that make containers real rather than nominal: they render, form-core
accepts them, the agent can READ ONE BACK (or the next turn flattens the form), and an op can
reach a field inside one.
"""
from __future__ import annotations

import pytest

from luke_agents.agents.form_agent.coltorapps import schema_to_spec, spec_to_schema
from luke_agents.agents.form_agent.ops import apply_operations
from luke_agents.agents.form_agent.schema import FormOp, FormSpec, SpecField


def _menu_spec() -> FormSpec:
    """The form the user actually asked for: a to-go order with tabbed menu sections."""
    return FormSpec(title="Savera To-Go", fields=[
        SpecField(key="customer", label="Your name", type="textField", required=True),
        SpecField(key="pickup", label="Pickup time", type="datetime"),
        SpecField(key="menu", label="Menu", type="tabs", children=[
            SpecField(key="appetizers", label="Appetizers", type="panel", children=[
                SpecField(key="samosa", label="Vegetable Samosa", type="stepper"),
                SpecField(key="pakora", label="Onion Pakora", type="stepper"),
            ]),
            SpecField(key="biryani", label="Biryani", type="panel", children=[
                SpecField(key="chicken_biryani", label="Chicken Biryani", type="stepper"),
            ]),
        ]),
        SpecField(key="place_order", label="Place Order", type="button"),
    ])


def _tree(fields, depth=0):
    out = []
    for f in fields:
        out.append(("  " * depth) + f"{f.key}:{f.type}")
        if f.children:
            out.extend(_tree(f.children, depth + 1))
    return out


# --------------------------------------------------------------------------- #
# Render: nesting reaches the schema in the shape form-core expects
# --------------------------------------------------------------------------- #
def test_a_container_renders_as_children_ids_with_parent_links():
    schema = spec_to_schema(_menu_spec())
    ents = schema["entities"]

    tabs = next(e for e in ents.values() if e["type"] == "tabs")
    panels = [ents[cid] for cid in tabs["children"]]
    assert [p["type"] for p in panels] == ["panel", "panel"]
    assert [p["attributes"]["label"] for p in panels] == ["Appetizers", "Biryani"]

    # The schema nests by REFERENCE, and `parentId` must agree with the parent's `children` —
    # form-core reports a mismatch as an error, so both halves have to be written.
    for cid in tabs["children"]:
        assert ents[cid]["parentId"] == next(k for k, v in ents.items() if v is tabs)

    # Only the top level appears in `root`; children are reached through their parent.
    assert len(schema["root"]) == 4
    for cid in tabs["children"]:
        assert cid not in schema["root"]


def test_a_container_carries_no_value_attributes():
    """A panel holds no answer. `required` on one is a schema form-core rejects, and a
    `placeholder` or `options` on a layout box is a setting nobody can ever see."""
    schema = spec_to_schema(FormSpec(fields=[
        SpecField(key="sec", label="Section", type="panel", required=True,
                  placeholder="nope", options=["a"], children=[
                      SpecField(key="x", label="X", type="textField"),
                  ]),
    ]))
    panel = next(e for e in schema["entities"].values() if e["type"] == "panel")
    for gone in ("required", "placeholder", "options"):
        assert gone not in panel["attributes"], f"{gone} must be stripped from a container"


# --------------------------------------------------------------------------- #
# Read back: the agent must SEE what it built, or the next turn destroys it
# --------------------------------------------------------------------------- #
def test_the_agent_can_read_its_own_nested_form_back():
    # The failure this prevents: build tabs, then next turn report the form has no tabs and be
    # unable to touch a single dish inside them, because the whole subtree was carried past the
    # model as opaque "preserved" entities.
    before = _menu_spec()
    schema = spec_to_schema(before)
    after, _existing, preserved, _roots = schema_to_spec(schema)

    assert preserved == {}, "nothing in this form should be invisible to the model"
    assert _tree(after.fields) == _tree(before.fields)


def test_round_tripping_twice_is_stable():
    """An edit is project → change → render. If that loop is not a fixpoint, a form drifts a
    little on every turn even when the user changes nothing."""
    once = spec_to_schema(_menu_spec())
    spec, existing, preserved, roots = schema_to_spec(once)
    twice = spec_to_schema(spec, existing=existing, preserved_entities=preserved,
                           preserved_root_ids=roots)

    def shape(s):
        return sorted(
            (e["type"], (e.get("attributes") or {}).get("key"), len(e.get("children") or []))
            for e in s["entities"].values()
        )

    assert shape(twice) == shape(once)


# --------------------------------------------------------------------------- #
# Ops: a field inside a tab is a field you can edit
# --------------------------------------------------------------------------- #
def test_a_dish_can_be_added_into_a_named_section():
    spec = apply_operations(_menu_spec(), [FormOp(
        op="add", parent="appetizers",
        field=SpecField(key="paneer_tikka", label="Paneer Tikka", type="stepper"),
    )])
    apps = next(f for f in spec.fields if f.key == "menu").children[0]
    assert [c.key for c in apps.children] == ["samosa", "pakora", "paneer_tikka"]


def test_a_nested_field_can_be_updated_and_removed_without_naming_its_parent():
    # `update`/`remove` search the whole tree: a top-level-only scan would miss the target, then
    # `update` would fall through to APPENDING a duplicate outside the container.
    spec = apply_operations(_menu_spec(), [FormOp(
        op="update", key="samosa",
        field=SpecField(key="samosa", label="Vegetable Samosa (2 pc)", type="stepper"),
    )])
    apps = next(f for f in spec.fields if f.key == "menu").children[0]
    assert [c.label for c in apps.children] == ["Vegetable Samosa (2 pc)", "Onion Pakora"]
    assert len(spec.fields) == 4, "the update must not leak a copy to the top level"

    spec = apply_operations(spec, [FormOp(op="remove", key="pakora")])
    apps = next(f for f in spec.fields if f.key == "menu").children[0]
    assert [c.key for c in apps.children] == ["samosa"]


def test_removing_a_container_takes_its_contents_with_it():
    spec = apply_operations(_menu_spec(), [FormOp(op="remove", key="menu")])
    assert [f.key for f in spec.fields] == ["customer", "pickup", "place_order"]


def test_renaming_a_container_does_not_empty_it():
    # A model that omits `children` on an update means "I did not touch them", not "delete them".
    spec = apply_operations(_menu_spec(), [FormOp(
        op="update", key="menu", field=SpecField(key="menu", label="Our Menu", type="tabs"),
    )])
    menu = next(f for f in spec.fields if f.key == "menu")
    assert menu.label == "Our Menu"
    assert [c.key for c in (menu.children or [])] == ["appetizers", "biryani"]


def test_reorder_applies_inside_a_named_container():
    spec = apply_operations(_menu_spec(), [FormOp(
        op="reorder", parent="appetizers", order=["pakora", "samosa"],
    )])
    apps = next(f for f in spec.fields if f.key == "menu").children[0]
    assert [c.key for c in apps.children] == ["pakora", "samosa"]
    # and the top level is untouched
    assert [f.key for f in spec.fields] == ["customer", "pickup", "menu", "place_order"]


def test_adding_into_something_that_is_not_a_container_falls_back_to_the_top_level():
    # Rather than dropping the field on the floor, which reads to the user as "it ignored me".
    spec = apply_operations(_menu_spec(), [FormOp(
        op="add", parent="customer",
        field=SpecField(key="notes", label="Notes", type="textarea"),
    )])
    assert "notes" in [f.key for f in spec.fields]


@pytest.mark.parametrize("ctype", ["panel", "fieldset", "well", "columns", "tabs", "table", "wizard", "page"])
def test_every_supported_container_round_trips(ctype):
    child = SpecField(key="inner", label="Inner", type="textField")
    spec = FormSpec(fields=[SpecField(key="box", label="Box", type=ctype, children=[child])])
    back, _e, preserved, _r = schema_to_spec(spec_to_schema(spec))
    assert preserved == {}
    assert back.fields[0].type == ctype
    assert [c.key for c in back.fields[0].children] == ["inner"]
