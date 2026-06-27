"""The agent can EMIT advanced field behaviour — conditional `logic` (show/hide/require/…),
`hidden`, `disabled`, `calculate_value` — and round-trips it through the coltorapps schema, so
LukeBuilds applies rules like "hide email when name is gowtham" instead of just explaining them."""
from luke_agents.agents.form_agent.coltorapps import schema_to_spec, spec_to_schema
from luke_agents.agents.form_agent.schema import FormSpec, SpecField, SpecLogicRule


def _field(**kw) -> SpecField:
    return SpecField(key=kw.pop("key", "f"), label=kw.pop("label", "F"), **kw)


def _attrs(schema: dict, key: str) -> dict:
    return next(e["attributes"] for e in schema["entities"].values() if e["attributes"]["key"] == key)


def test_logic_hide_rule_is_written_to_attributes():
    spec = FormSpec(fields=[
        _field(key="name", label="Name"),
        _field(key="email", label="Email", type="email",
               logic=[SpecLogicRule(when='name == "gowtham"', action="hide")]),
    ])
    schema = spec_to_schema(spec)
    assert _attrs(schema, "email")["logic"] == [{"when": 'name == "gowtham"', "action": "hide"}]


def test_hidden_disabled_calculate_round_trip():
    schema = spec_to_schema(FormSpec(fields=[
        _field(key="total", label="Total", type="number",
               hidden=True, disabled=True, calculate_value="qty * price"),
    ]))
    attrs = _attrs(schema, "total")
    assert attrs["hidden"] is True and attrs["disabled"] is True
    assert attrs["calculateValue"] == "qty * price"
    # Reading it back surfaces the same advanced settings to the LLM.
    f = schema_to_spec(schema)[0].fields[0]
    assert f.hidden is True and f.disabled is True and f.calculate_value == "qty * price"


def test_calculate_value_skipped_on_button():
    schema = spec_to_schema(FormSpec(fields=[_field(key="submit", label="Submit", type="button", calculate_value="1")]))
    assert "calculateValue" not in _attrs(schema, "submit")  # a button has no value


def test_existing_logic_survives_an_unrelated_edit():
    base = spec_to_schema(FormSpec(fields=[
        _field(key="email", label="Email", type="email",
               logic=[SpecLogicRule(when='name == "x"', action="hide")]),
    ]))
    spec, existing, pe, pr = schema_to_spec(base)
    spec.fields[0].label = "Email address"  # change something unrelated
    out = spec_to_schema(spec, existing, pe, pr)
    attrs = _attrs(out, "email")
    assert attrs["label"] == "Email address"
    assert attrs["logic"] == [{"when": 'name == "x"', "action": "hide"}]


def test_malformed_logic_is_dropped_not_fatal():
    # A hand-edited schema with a bad rule must not break the projection.
    schema = {"entities": {"e1": {"type": "textField", "attributes": {
        "key": "x", "label": "X", "logic": [{"action": "bogus"}, {"when": "a == 1", "action": "hide"}],
    }}}, "root": ["e1"]}
    f = schema_to_spec(schema)[0].fields[0]
    assert f.logic is not None and len(f.logic) == 1 and f.logic[0].action == "hide"
