"""Apply the LLM's targeted operations to the current form.

This is what makes edits surgical: the model emits one op per thing the user
asked to change, and we apply just those onto the existing field list. Any field
the model didn't mention is carried through untouched — and because
spec_to_schema merges onto the existing entity by key, those fields stay
byte-for-byte identical (same id, same attributes) in the rendered schema.
"""
from __future__ import annotations

from typing import List

from .schema import FormOp, FormSpec, SpecField


def apply_operations(current: FormSpec, operations: List[FormOp]) -> FormSpec:
    fields: list[SpecField] = list(current.fields)
    title = current.title

    def index_of(key: str | None) -> int:
        if not key:
            return -1
        for i, f in enumerate(fields):
            if f.key == key:
                return i
        return -1

    for op in operations:
        if op.op == "retitle":
            if op.title is not None:
                title = op.title

        elif op.op == "add":
            if op.field is None:
                continue
            existing = index_of(op.field.key)
            if existing >= 0:
                fields[existing] = op.field  # key already present → treat as in-place replace
            else:
                after = index_of(op.after)
                if after >= 0:
                    fields.insert(after + 1, op.field)
                else:
                    fields.append(op.field)

        elif op.op == "update":
            if op.field is None:
                continue
            # Target the explicit key if given, else the field's own key. Keep
            # position; replacing in place preserves the entity id on render
            # (spec_to_schema matches by key) as long as the key is unchanged.
            target = index_of(op.key) if op.key else index_of(op.field.key)
            if target >= 0:
                fields[target] = op.field
            else:
                fields.append(op.field)  # unknown target → add it

        elif op.op == "remove":
            target = index_of(op.key or (op.field.key if op.field else None))
            if target >= 0:
                del fields[target]

        elif op.op == "reorder":
            if op.order:
                wanted = [k for k in op.order if index_of(k) >= 0]
                seen = set(wanted)
                rest = [f.key for f in fields if f.key not in seen]  # keep unmentioned at the end
                by_key = {f.key: f for f in fields}
                fields = [by_key[k] for k in (wanted + rest)]

    return FormSpec(title=title, fields=fields)
