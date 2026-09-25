"""Convert between the LLM's flat FormSpec and the coltorapps builder schema.

coltorapps schema shape (what luke-consumer-ui's FormRenderer / builderStore and
luke-capability-engine's draftSchema expect):

    {
      "entities": {
        "<id>": { "type": "textField", "attributes": { "label": ..., "key": ..., "required": ... } },
        "<id>": { "type": "panel", "attributes": {...}, "children": ["<id>", ...] },
        ...
      },
      "root": ["<id>", "<id>", ...]   # top-level order
    }

The submission key for a field is `attributes.key` (falling back to the entity
id). We always set `key` explicitly so data keys are stable across edits.

We only expose flat, top-level "simple" fields to the LLM. Everything else
(containers, their nested children, and advanced field types we don't model) is
preserved verbatim so AI edits never corrupt a complex form. Preserved
top-level entities are kept after the simple fields in `root`.
"""
from __future__ import annotations

import re
import unicodedata
import uuid

from typing import get_args

from .schema import (CHOICE_TYPES, CONTAINER_TYPES, STATIC_TYPES, FieldType, FormSpec,
                     SpecField)

# DERIVED, never re-listed. This was a second hand-written copy of the same set as
# `FieldType`, and the two drifted apart exactly as you would expect: on 2026-09-25 both still
# held 13 types while form-core offered 48, so the agent told a user the builder had "no layout
# containers like tabs or panels and no dedicated stepper control" while they were looking at all
# three. Adding a type in one place and not the other is the whole bug; there is now one place.
# `FieldType` in turn is held to form-core's registry by tests/test_field_type_parity.py.
KNOWN_FIELD_TYPES = set(get_args(FieldType))

# The standard Lukeflow geocoding provider for the structured address field. An `addressBlock`
# becomes a type-ahead autocomplete (street → fills city/region/postal/country) when it names a
# `dataSource` minion; we attach the platform default so the LLM only has to choose the TYPE.
ADDRESS_DATA_SOURCE = {"minion": "geocode"}

# Types whose coltorapps definition includes `placeholderAttribute`. Setting
# `placeholder` on any other type produces an "Unknown entity attribute" schema.
PLACEHOLDER_TYPES = {
    "textField", "textarea", "number", "email", "phoneNumber", "url", "password",
    "currency", "select", "searchSelect", "tags",
}

# Types that collect no answer, so `required` / `placeholder` / `options` / help text are
# meaningless on them — a `required` heading is a schema form-core rejects. `button` was the
# original member; the other statics arrived with the rest of the registry.
NO_REQUIRED_TYPES = set(STATIC_TYPES) | set(CONTAINER_TYPES)


# ── Data-key safety ───────────────────────────────────────────────────────────
# form-core validates every submission key against this exact identifier rule and rejects
# violations as ERRORS (blocking check-in and publish), so a schema we emit with a bad key is a
# form the author cannot ship. See @lukeflow/form-core KEY_REGEX_SOURCE / RESERVED_KEYS — keep
# these two constants in step with it; fixtures/agent-schema-cases.json is the cross-language
# guard that they have not drifted.
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_KEYS = {"true", "false", "null", "undefined", "NaN", "Infinity"}


def _safe_key(raw: object, label: object, taken: set[str]) -> str:
    """Coerce anything into a UNIQUE, valid form-core data key.

    The LLM is *asked* for snake_case, which is not the same as being held to it: it will
    eventually answer "Full Name", "e-mail", "naïve" or "123". Each of those renders a schema
    form-core rejects outright, so the author gets an AI-built form that cannot be checked in.
    Normalising here — the single point where every schema is produced — means no upstream path
    (first build, chat edit, hand-edited import) can emit an unusable key.
    """
    text = str(raw or "").strip()
    # Latin accents decompose to ASCII (naïve -> naive) rather than being dropped to nothing.
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    text = re.sub(r"_{2,}", "_", text)

    if not text or text[0].isdigit():
        # Fall back to the LABEL before an anonymous name: "123" on a field called "Age" is far
        # more useful to a human reading the payload as "age" than as "field".
        from_label = re.sub(r"[^A-Za-z0-9_]+", "_",
                            unicodedata.normalize("NFKD", str(label or "")).encode("ascii", "ignore").decode("ascii")).strip("_")
        if from_label and not from_label[0].isdigit():
            # Lowercased because this key is SYNTHESISED by us, and SpecField documents keys as
            # snake_case. A key the author actually chose keeps its casing (firstName stays
            # firstName) — only our invention follows the house style.
            text = re.sub(r"_{2,}", "_", from_label).lower()
        elif text:
            text = f"field_{text}"  # keep the digits, just make them addressable
        else:
            text = "field"

    if text in _RESERVED_KEYS or not _KEY_RE.match(text):
        text = f"{text}_"

    # Uniqueness is checked across the WHOLE form, including entities the LLM never sees (a
    # container's nested children), so this must consider preserved keys too — see spec_to_schema.
    if text not in taken:
        taken.add(text)
        return text
    n = 2
    while f"{text}_{n}" in taken:
        n += 1
    out = f"{text}_{n}"
    taken.add(out)
    return out


def _collect_keys(entities: dict) -> set[str]:
    """Every data key already spoken for by a preserved entity, at any depth."""
    out: set[str] = set()
    for ent in (entities or {}).values():
        if isinstance(ent, dict):
            k = (ent.get("attributes") or {}).get("key")
            if isinstance(k, str) and k:
                out.add(k)
    return out


def _new_id() -> str:
    # coltorapps validates entity ids as canonical UUIDs (8-4-4-4-12); a
    # truncated hex string is rejected by validateEntityId.
    return str(uuid.uuid4())


_LOGIC_ACTIONS = {"show", "hide", "enable", "disable", "require", "optional", "setValue"}


def _safe_logic(raw: object) -> list | None:
    """Read existing logic rules leniently — keep only well-formed {when, action[, value]} so a
    hand-edited schema can't break the projection (and the LLM still sees the valid rules)."""
    if not isinstance(raw, list):
        return None
    out: list = []
    for r in raw:
        if isinstance(r, dict) and r.get("action") in _LOGIC_ACTIONS:
            rule = {"when": str(r.get("when") or ""), "action": r["action"]}
            if r.get("value") is not None:
                rule["value"] = str(r["value"])
            out.append(rule)
    return out or None


def schema_to_spec(schema: dict | None) -> tuple[FormSpec, dict, dict, list]:
    """Project a coltorapps schema down to a flat FormSpec the LLM can edit, plus
    the bookkeeping needed to rebuild without losing anything:

      - existing: {key -> {"id","type","attributes"}} for simple fields, so edits
        merge onto (preserve) the original attributes / entity id.
      - preserved_entities: {id -> entity} for EVERY entity that isn't a rebuilt
        simple field — containers AND their nested children — carried verbatim.
      - preserved_root_ids: the top-level ids among those, kept for `root`.
    """
    schema = schema or {}
    entities = schema.get("entities", {}) or {}
    root = schema.get("root") or list(entities.keys())

    fields: list[SpecField] = []
    existing: dict = {}
    simple_ids: set = set()
    preserved_root_ids: list = []

    def project(eid) -> SpecField | None:
        """One entity as a SpecField, recursing into a container's children — or None when this
        is something the agent cannot represent, which the caller then preserves verbatim.

        Recursion is what makes an edit possible at all. Carried through untouched instead, a
        container and everything in it is INVISIBLE to the model: it would build tabs, then on the
        very next turn report that the form has no tabs and be unable to touch a single item
        inside them."""
        ent = entities.get(eid)
        if not isinstance(ent, dict):
            return None
        etype = ent.get("type", "")
        attrs = ent.get("attributes", {}) or {}
        kids = ent.get("children") or []

        if etype in CONTAINER_TYPES:
            projected = [project(cid) for cid in kids]
            if any(p is None for p in projected):
                return None  # something inside is unrepresentable — keep the WHOLE subtree intact
            key = str(attrs.get("key") or eid)
            existing[key] = {"id": eid, "type": etype, "attributes": dict(attrs)}
            simple_ids.add(eid)
            return SpecField(
                key=key,
                label=str(attrs.get("label", key)),
                type=etype,  # type: ignore[arg-type]
                children=projected,  # type: ignore[arg-type]
                hidden=attrs.get("hidden") if isinstance(attrs.get("hidden"), bool) else None,
                logic=_safe_logic(attrs.get("logic")),
            )

        if etype not in KNOWN_FIELD_TYPES or kids:
            return None  # unknown type, or a container we do not model: preserve it

        key = str(attrs.get("key") or eid)
        opts = attrs.get("options")
        existing[key] = {"id": eid, "type": etype, "attributes": dict(attrs)}
        simple_ids.add(eid)
        return SpecField(
            key=key,
            label=str(attrs.get("label", key)),
            type=etype,  # type: ignore[arg-type]
            required=bool(attrs.get("required", False)),
            options=list(opts) if isinstance(opts, list) else None,
            placeholder=attrs.get("placeholder"),
            tooltip=attrs.get("tooltip"),
            description=attrs.get("description"),
            hidden=attrs.get("hidden") if isinstance(attrs.get("hidden"), bool) else None,
            disabled=attrs.get("disabled") if isinstance(attrs.get("disabled"), bool) else None,
            logic=_safe_logic(attrs.get("logic")),
            calculate_value=attrs.get("calculateValue") if isinstance(attrs.get("calculateValue"), str) else None,
        )

    for eid in root:
        field = project(eid)
        if field is None:
            preserved_root_ids.append(eid)
        else:
            fields.append(field)

    # Preserve every entity that isn't a rebuilt simple field — this includes
    # containers AND their nested children (which never appear in `root`).
    preserved_entities = {
        eid: ent for eid, ent in entities.items() if eid not in simple_ids
    }

    return FormSpec(fields=fields), existing, preserved_entities, preserved_root_ids


def spec_to_schema(
    spec: FormSpec,
    existing: dict | None = None,
    preserved_entities: dict | None = None,
    preserved_root_ids: list | None = None,
) -> dict:
    """Render a FormSpec into a coltorapps schema. Fields whose key matches an
    existing entity reuse its id and merge onto its attributes (so advanced
    builder settings survive). Preserved entities are carried through verbatim;
    preserved top-level ones are appended after the simple fields in `root`."""
    existing = existing or {}
    preserved_entities = dict(preserved_entities or {})
    preserved_root_ids = list(preserved_root_ids or [])

    entities: dict = {}
    root: list = []
    used: set = set()
    # Keys already spoken for by entities the LLM cannot see — a container's nested children share
    # the submission namespace, so a new top-level field named `inner` would silently collide with
    # a panel child of the same name and form-core would reject BOTH as duplicate-key.
    taken_keys: set[str] = _collect_keys(preserved_entities)

    def emit(f, parent_id=None) -> str:
        """Render one field (and, for a container, everything inside it) and return its id.

        Recursive because a container is a field that holds fields: the schema nests by ID —
        `children: [id]` on the parent, `parentId` on each child — so a subtree is emitted
        bottom-up and wired by reference, never by embedding."""
        # Look the previous entity up by the key the LLM used, BEFORE normalising: that is the key
        # it saw in the projection, and matching on it is what preserves the entity id and its
        # advanced attributes across an edit.
        prev = existing.get(f.key)
        eid = prev["id"] if prev else _new_id()
        # Never collide with a preserved entity id or one already emitted.
        while eid in preserved_entities or eid in used:
            eid = _new_id()
        used.add(eid)

        # Merge onto the prior attributes ONLY when the type is unchanged — a
        # different type has a different (incompatible) attribute set, so reusing
        # e.g. textField's minLength on a select would be an invalid attribute.
        same_type = bool(prev) and prev.get("type") == f.type
        attrs = dict(prev["attributes"]) if same_type else {}

        attrs["label"] = f.label
        # NORMALISED, never the raw value: form-core rejects a key that isn't a plain identifier
        # (and rejects duplicates) as an ERROR, which blocks check-in and publish outright.
        attrs["key"] = _safe_key(f.key, f.label, taken_keys)

        if f.type in NO_REQUIRED_TYPES:
            attrs.pop("required", None)
        else:
            attrs["required"] = f.required

        if f.type in PLACEHOLDER_TYPES and f.placeholder is not None:
            attrs["placeholder"] = f.placeholder
        elif f.type not in PLACEHOLDER_TYPES:
            attrs.pop("placeholder", None)  # strip if carried from a prior type

        # Help text — tooltip (hover ⓘ) and description (line below the field).
        # Every type except button supports both; button has no such attributes.
        for help_attr, val in (("tooltip", f.tooltip), ("description", f.description)):
            if f.type in NO_REQUIRED_TYPES:
                attrs.pop(help_attr, None)
            elif val is not None:
                attrs[help_attr] = val

        if f.type in CHOICE_TYPES:
            attrs["options"] = f.options or ["Option 1"]
        else:
            attrs.pop("options", None)

        # A structured address field autocompletes via the platform geocoding provider. Default it
        # in (preserving any provider already configured) so the LLM only chooses type=addressBlock.
        if f.type == "addressBlock":
            attrs.setdefault("dataSource", dict(ADDRESS_DATA_SOURCE))
        else:
            attrs.pop("dataSource", None)  # strip if carried from a prior type

        # Advanced behaviours — set when provided (None = leave as-is; a same-type merge already
        # carried any prior value). hidden/disabled/logic apply to any field; calculateValue is
        # meaningless on a button (it has no value), so skip it there.
        if f.hidden is not None:
            attrs["hidden"] = f.hidden
        if f.disabled is not None:
            attrs["disabled"] = f.disabled
        if f.logic is not None:
            attrs["logic"] = [r.model_dump(exclude_none=True) for r in f.logic]
        if f.calculate_value is not None and f.type not in NO_REQUIRED_TYPES:
            attrs["calculateValue"] = f.calculate_value

        ent = {"type": f.type, "attributes": attrs}
        if parent_id is not None:
            ent["parentId"] = parent_id
        # A container holds no answer, so the value-shaped attributes above are meaningless on it;
        # `required` on a panel in particular makes form-core reject the schema.
        if f.type in CONTAINER_TYPES:
            for gone in ("required", "placeholder", "options", "dataSource", "calculateValue"):
                ent["attributes"].pop(gone, None)
            entities[eid] = ent  # registered BEFORE the children, so a child can point back at it
            ent["children"] = [emit(c, eid) for c in (f.children or [])]
        else:
            entities[eid] = ent
        return eid

    for f in spec.fields:
        root.append(emit(f))

    # Carry every preserved entity through unchanged (containers + their children).
    for eid, ent in preserved_entities.items():
        entities[eid] = ent
    # Keep preserved top-level entities in `root`, after the simple fields.
    for eid in preserved_root_ids:
        if eid in preserved_entities:
            root.append(eid)

    return {"entities": entities, "root": root}
