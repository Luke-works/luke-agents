"""Regenerate fixtures/agent-schema-cases.json — the rendered schemas luke-forms validates.

    python -m tests.regen_fixture      # then copy the file into luke-forms/fixtures/

REVISION is a tripwire, not decoration: luke-forms pins the value it expects, so a regenerated
fixture that never reaches that repo fails there loudly instead of silently checking stale cases.
Bump it whenever the rendered output changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.form_matrix import build_cases

REVISION = "2026-08-01.1"

def main() -> None:
    out = Path(__file__).resolve().parents[1] / "fixtures" / "agent-schema-cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "Schemas rendered by luke-agents' form agent (tests/form_matrix.py), validated in "
            "luke-forms by form-core's real validateSchema. Regenerate with "
            "`python -m tests.regen_fixture` and copy into luke-forms/fixtures/."
        ),
        "revision": REVISION,
        "cases": build_cases(),
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(payload['cases'])} cases to {out} (revision {REVISION})")

if __name__ == "__main__":
    main()
