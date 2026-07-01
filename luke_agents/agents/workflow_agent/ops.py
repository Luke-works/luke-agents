"""Server-side sanitize/repair for the WorkflowDoc the LLM returns.

This is the workflow agent's equivalent of workflow-core's `repairWorkflow`: before
we hand a doc back to the UI we dedupe node ids and rewrite every dangling reference
(a `next`/`else`/`join`/branch target that points at a non-existent node) to the
terminal "end" — so the builder ALWAYS gets a valid, wireable graph no matter what
the model emitted. The client also re-runs `repairWorkflow`/`validateWorkflow`, so
this is defense-in-depth, not the sole guarantee.

It also derives the small conversational bits (`reply`, `suggestions`) the chat
response needs but that the WorkflowDoc response_model doesn't carry.
"""
from __future__ import annotations

from typing import List

from .schema import END_NODE, WorkflowDocModel, WorkflowNodeModel


def _fix(target: object, valid: set[str]) -> object:
    """Keep a target only if it names an existing node or is the terminal sentinel;
    otherwise route it to "end". Absent (None) is left absent (also terminal)."""
    if target is None:
        return None
    if target == END_NODE or target in valid:
        return target
    return END_NODE


def repair_doc(doc: WorkflowDocModel) -> WorkflowDocModel:
    """Dedupe ids and repair dangling references so the graph is always wireable."""
    # Dedupe by id (first-seen wins), dropping blank ids defensively.
    kept: List[WorkflowNodeModel] = []
    seen_ids: set[str] = set()
    for n in doc.nodes:
        if not n.id or n.id in seen_ids:
            continue
        seen_ids.add(n.id)
        kept.append(n)

    valid = set(seen_ids)
    for n in kept:
        n.next = _fix(n.next, valid)  # type: ignore[assignment]
        n.join = _fix(n.join, valid)  # type: ignore[assignment]
        n.else_ = _fix(n.else_, valid)  # type: ignore[assignment]
        if n.branches is not None:
            n.branches = [t for t in (_fix(b, valid) for b in n.branches) if t]  # type: ignore[misc]
        if n.conditions is not None:
            for c in n.conditions:
                c.next = _fix(c.next, valid) or END_NODE  # type: ignore[assignment]

    start = doc.start if (doc.start in valid) else None
    return WorkflowDocModel(
        id=doc.id or "wf",
        version=doc.version or 1,
        name=doc.name,
        trigger=doc.trigger,
        nodes=kept,
        start=start,
    )


def dump_doc(doc: WorkflowDocModel) -> dict:
    """Serialize to the portable JSON shape: reserved word `else` (not `else_`) and
    no null keys, so the emitted doc matches the workflow-core TypeScript optionals."""
    return doc.model_dump(by_alias=True, exclude_none=True)


def derive_reply(changed: bool, doc: WorkflowDocModel) -> str:
    """A short, friendly natural-language reply for the chat response (the LLM
    returns the doc only, so we phrase the reply deterministically)."""
    if not changed:
        return "Got it — I left your workflow as-is. Tell me what you'd like to change."
    n = len(doc.nodes)
    count = "1 step" if n == 1 else f"{n} steps"
    return f"Done — your workflow now has {count}. Want me to wire up anything else?"


def derive_suggestions(doc: WorkflowDocModel) -> List[str]:
    """A few contextual next-step ideas, based on what the workflow has so far."""
    kinds = {n.kind for n in doc.nodes}
    caps = {n.capability for n in doc.nodes if n.capability}
    out: List[str] = []
    if not doc.nodes:
        return ["Email the customer when the form is submitted", "Add an approval task", "Branch on a field value"]
    if "email" not in caps:
        out.append("Send a confirmation email")
    if "task" not in kinds:
        out.append("Add a human approval step")
    if "branch" not in kinds:
        out.append("Branch on a field value")
    if "integrations" not in caps:
        out.append("Sync the result to Salesforce")
    out.append("Add a delay before the next step")
    return out[:4]
