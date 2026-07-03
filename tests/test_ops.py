"""Unit tests for apply_operations — the surgical form-edit logic that applies the
LLM's targeted ops onto the current form (fields it doesn't mention stay intact)."""
from luke_agents.agents.form_agent.ops import apply_operations
from luke_agents.agents.form_agent.schema import FormSpec, SpecField, FormOp


def _f(key, label="L"):
    return SpecField(key=key, label=label)


def test_add_appends_when_no_anchor():
    out = apply_operations(FormSpec(fields=[_f("a")]), [FormOp(op="add", field=_f("b"))])
    assert [x.key for x in out.fields] == ["a", "b"]


def test_add_inserts_after_anchor():
    out = apply_operations(
        FormSpec(fields=[_f("a"), _f("c")]),
        [FormOp(op="add", field=_f("b"), after="a")],
    )
    assert [x.key for x in out.fields] == ["a", "b", "c"]


def test_update_replaces_in_place():
    out = apply_operations(
        FormSpec(fields=[_f("a", "old"), _f("b")]),
        [FormOp(op="update", field=_f("a", "new"))],
    )
    assert [x.key for x in out.fields] == ["a", "b"]
    assert out.fields[0].label == "new"


def test_remove_deletes_by_key():
    out = apply_operations(
        FormSpec(fields=[_f("a"), _f("b")]), [FormOp(op="remove", key="b")]
    )
    assert [x.key for x in out.fields] == ["a"]


def test_retitle_changes_only_title():
    out = apply_operations(FormSpec(title="Old", fields=[_f("a")]), [FormOp(op="retitle", title="New")])
    assert out.title == "New"
    assert [x.key for x in out.fields] == ["a"]


def test_unmentioned_fields_are_untouched():
    out = apply_operations(
        FormSpec(fields=[_f("a"), _f("b"), _f("c")]),
        [FormOp(op="update", field=_f("b", "changed"))],
    )
    assert [x.key for x in out.fields] == ["a", "b", "c"]
    assert out.fields[1].label == "changed"
    assert out.fields[0].label == "L" and out.fields[2].label == "L"
