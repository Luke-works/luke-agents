"""Apply the LLM's targeted operations to the current form.

This is what makes edits surgical: the model emits one op per thing the user
asked to change, and we apply just those onto the existing field list. Any field
the model didn't mention is carried through untouched — and because
spec_to_schema merges onto the existing entity by key, those fields stay
byte-for-byte identical (same id, same attributes) in the rendered schema.
"""
from __future__ import annotations

import typing
from typing import List

from .schema import CONTAINER_TYPES, FieldType, FormOp, FormSpec, SpecField

# The exact set of field types we know how to render (the coltorapps palette). Pydantic
# already constrains `SpecField.type` to this Literal, but we re-derive the allowlist and
# re-check at the apply boundary as defense in depth (#26): a model that emits an op with
# an unknown type/op is rejected loudly rather than silently rendered.
_ALLOWED_FIELD_TYPES = set(typing.get_args(FieldType))
_ALLOWED_OPS = {"add", "update", "remove", "reorder", "retitle"}


class UnsupportedOperation(ValueError):
    """An LLM-produced operation referenced an op kind or field type outside the allowlist."""


def validate_operations(operations: List[FormOp]) -> None:
    """Reject ops referencing unknown op kinds or field types before we apply them.

    Pydantic validation already runs on the parsed model; this is a second, explicit
    gate right at the apply callsite so a schema drift (or a coaxed-malformed field)
    can't slip an unrenderable type into the form."""
    for op in operations:
        if op.op not in _ALLOWED_OPS:
            raise UnsupportedOperation(f"unsupported op kind: {op.op!r}")
        field = op.field
        if field is not None and field.type not in _ALLOWED_FIELD_TYPES:
            raise UnsupportedOperation(f"unsupported field type: {field.type!r}")


def apply_operations(current: FormSpec, operations: List[FormOp]) -> FormSpec:
    validate_operations(operations)
    fields: list[SpecField] = list(current.fields)
    title = current.title

    def locate(key: str | None, where: list[SpecField] | None = None):
        """Find `key` ANYWHERE in the tree, as the (list, index) that holds it.

        A flat scan of the top level was enough while the agent could not build containers. Now
        that it can, a field inside a tab is a field the model must be able to edit: a top-level
        search would miss it, `update` would fall through to appending a duplicate outside the
        container, and `remove` would quietly do nothing."""
        if not key:
            return None
        where = fields if where is None else where
        for i, f in enumerate(where):
            if f.key == key:
                return where, i
            if f.children:
                hit = locate(key, f.children)
                if hit is not None:
                    return hit
        return None

    def container_children(key: str | None) -> list[SpecField] | None:
        """The child list of the container named `key`, or None for "the top level"."""
        if not key:
            return None
        hit = locate(key)
        if hit is None:
            return None
        holder, i = hit
        parent = holder[i]
        if parent.type not in CONTAINER_TYPES:
            return None  # asked to nest inside something that cannot hold fields
        if parent.children is None:
            parent.children = []
        return parent.children

    for op in operations:
        if op.op == "retitle":
            if op.title is not None:
                title = op.title

        elif op.op == "add":
            if op.field is None:
                continue
            existing = locate(op.field.key)
            if existing is not None:
                holder, i = existing
                holder[i] = op.field  # key already present → treat as in-place replace
            else:
                after = locate(op.after)
                if after is not None:
                    holder, i = after
                    holder.insert(i + 1, op.field)
                else:
                    # `parent` names the container to drop it into; absent — or naming something
                    # that cannot hold fields — it goes to the top level, which is what every
                    # op meant before containers existed.
                    into = container_children(op.parent)
                    (fields if into is None else into).append(op.field)

        elif op.op == "update":
            if op.field is None:
                continue
            # Target the explicit key if given, else the field's own key. Keep
            # position; replacing in place preserves the entity id on render
            # (spec_to_schema matches by key) as long as the key is unchanged.
            target = locate(op.key) if op.key else locate(op.field.key)
            if target is not None:
                holder, i = target
                # Carry the existing children across when the model omits them: an `update` that
                # renames a panel must not empty it.
                if op.field.children is None and holder[i].children and op.field.type in CONTAINER_TYPES:
                    op.field.children = holder[i].children
                holder[i] = op.field
            else:
                fields.append(op.field)  # unknown target → add it

        elif op.op == "remove":
            target = locate(op.key or (op.field.key if op.field else None))
            if target is not None:
                holder, i = target
                del holder[i]  # a container goes with everything inside it, as the person expects

        elif op.op == "reorder":
            if op.order:
                # Reorder within ONE list: the top level, or the named container's children.
                scope = container_children(op.parent) if op.parent else fields
                if scope is None:
                    scope = fields
                by_key = {f.key: f for f in scope}
                wanted = [k for k in op.order if k in by_key]
                seen = set(wanted)
                rest = [f.key for f in scope if f.key not in seen]  # unmentioned keep their place
                ordered = [by_key[k] for k in (wanted + rest)]
                if scope is fields:
                    fields = ordered
                else:
                    scope[:] = ordered

    return FormSpec(title=title, fields=fields)
