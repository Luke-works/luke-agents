"""WorkflowDoc schema — the workflow agent's LLM contract.

The LLM emits a BOUNDED `WorkflowDoc` (a graph of nodes) each turn — the SAME
portable JSON `@lukeflow/workflow-core` authors, `luke-consumer-ui` renders in the
react-flow builder, and `luke-core-engine` compiles to BPMN. The shape is
intentionally permissive-but-bounded: nodes carry a `kind` and (for action/task) a
`capability`, and the server repairs dangling references before returning, so the
UI always gets a valid, wireable document no matter what the model emitted.

Mirrors `email_agent` behind the `Agent` contract: full doc per turn, repaired
server-side. Kept deliberately loose per-node (all fields optional but `id`+`kind`)
so a slightly-off node never 502s the whole turn — the reserved terminal target is
the string ``"end"``.
"""
from __future__ import annotations

import json
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Bound request size so a single (paid) call can't amplify cost or exhaust memory.
_MAX_MESSAGE_CHARS = 8_000
_MAX_DOC_BYTES = 300_000
_MAX_NODES = 60

# The closed set of executable node kinds (matches workflow-core `NodeKind`).
_NODE_KINDS = {"action", "task", "branch", "parallel", "wait"}
# The reserved terminal target (matches workflow-core `END_NODE`).
END_NODE = "end"


def _validate_doc_size(v: Optional[dict]) -> Optional[dict]:
    if v is not None and len(json.dumps(v, default=str)) > _MAX_DOC_BYTES:
        raise ValueError("doc is too large")
    return v


class TriggerModel(BaseModel):
    """How the workflow starts — a capability's inbound Trigger."""
    capability: str = Field(default="forms", description="e.g. 'forms', 'email', 'integrations'.")
    type: str = Field(default="form.submitted", description="Event type, e.g. 'form.submitted'.")
    config: Optional[dict] = Field(default=None, description="Trigger-specific data, e.g. {formId}.")


class BranchConditionModel(BaseModel):
    """One arm of a branch: take `next` when `expr` is truthy."""
    expr: str = Field(default="", description="Boolean expression over process vars, e.g. amount > 10000.")
    next: str = Field(default=END_NODE, description="Target node id, or 'end'.")


class WorkflowNodeModel(BaseModel):
    """A single graph node. Only `id` + `kind` are required; the rest depend on the
    kind. Unknown fields are ignored (the compiler tolerates extras)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(description="Unique node id within the workflow, e.g. 'n1'.")
    kind: str = Field(description="action | task | branch | parallel | wait.")
    name: Optional[str] = Field(default=None, description="Optional human label.")

    # action / task
    capability: Optional[str] = Field(default=None, description="Capability id for action/task nodes.")
    action: Optional[str] = Field(default=None, description="Action op for kind=action, e.g. 'send'.")
    task: Optional[str] = Field(default=None, description="Task type for kind=task, e.g. 'review'.")
    provider: Optional[str] = Field(default=None, description="For integrations: the connector, e.g. 'salesforce'.")
    connection: Optional[str] = Field(default=None, description="Connection selector for integration actions.")
    input: Optional[dict] = Field(default=None, description="Input map bound to the action/task inputs.")
    output: Optional[str] = Field(default=None, description="Process variable to store the result under.")
    assignee: Optional[str] = Field(default=None, description="Assignee expression for tasks, e.g. 'queue:ops'.")
    on_error: Optional[dict] = Field(default=None, alias="onError", description="Error policy (retry/fallback).")

    # linear successor (action / task / wait)
    next: Optional[str] = Field(default=None, description="Successor node id, or 'end'/absent to terminate.")

    # branch
    conditions: Optional[List[BranchConditionModel]] = Field(default=None, description="Branch arms.")
    else_: Optional[str] = Field(default=None, alias="else", description="Branch default target, or 'end'.")

    # parallel
    branches: Optional[List[str]] = Field(default=None, description="Concurrent branch target node ids.")
    join: Optional[str] = Field(default=None, description="Node to continue at after all branches complete.")

    # wait
    mode: Optional[str] = Field(default=None, description="'timer' or 'event' for kind=wait.")
    duration: Optional[str] = Field(default=None, description="ISO/human duration for a timer wait, e.g. 'P1D'.")
    event: Optional[dict] = Field(default=None, description="Inbound event to correlate on for an event wait.")


class WorkflowDocModel(BaseModel):
    """The full workflow document the LLM returns each turn (max 60 nodes)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default="wf", description="Workflow id.")
    version: int = Field(default=1, description="Monotonic definition version.")
    name: Optional[str] = Field(default=None, description="Human name.")
    trigger: TriggerModel = Field(default_factory=TriggerModel)
    nodes: List[WorkflowNodeModel] = Field(default_factory=list, description="Graph nodes (max 60).")
    start: Optional[str] = Field(default=None, description="Explicit start node id; defaults to nodes[0].")

    @field_validator("nodes", mode="before")
    @classmethod
    def _drop_bad_nodes(cls, v: object) -> object:
        """Server-side repair: drop entries that aren't objects or whose `kind` is
        outside the bounded vocabulary, so one stray node never 502s the turn."""
        if not isinstance(v, list):
            return v
        return [
            n for n in v
            if isinstance(n, BaseModel)
            or (isinstance(n, dict) and n.get("kind") in _NODE_KINDS and isinstance(n.get("id"), str) and n.get("id"))
        ]

    @field_validator("nodes")
    @classmethod
    def _cap_nodes(cls, v: List[WorkflowNodeModel]) -> List[WorkflowNodeModel]:
        if len(v) > _MAX_NODES:
            raise ValueError(f"too many nodes (max {_MAX_NODES})")
        return v


class ChatRequest(BaseModel):
    """One turn. Stateless: the client (the Workflow Builder) sends the current
    WorkflowDoc back each time, so the agent needs no storage."""
    message: str = Field(max_length=_MAX_MESSAGE_CHARS)
    # Current WorkflowDoc, or null/absent for a brand-new workflow.
    doc: Optional[dict] = None
    title: Optional[str] = Field(default=None, max_length=500)
    # The step-type catalog (GET /api/workflow/catalog) so the model only uses
    # capabilities the tenant actually has. Optional — a sensible default set is
    # used when absent.
    catalog: Optional[List[dict]] = None
    user_id: Optional[str] = Field(default=None, max_length=200)  # advisory only; budget is keyed by IP
    session_id: Optional[str] = Field(default=None, max_length=200)
    consent: bool = True  # may this turn be retained for model fine-tuning?

    @field_validator("doc")
    @classmethod
    def _cap_doc(cls, v: Optional[dict]) -> Optional[dict]:
        return _validate_doc_size(v)


class ChatResponse(BaseModel):
    doc: dict  # full updated WorkflowDoc, ready for the builder / updateDraft
    title: str
    reply: str = ""  # natural-language message to show the user
    suggestions: List[str] = []  # clickable next-step ideas
    changed: bool = True  # False when the workflow was untouched (e.g. a question)
    brain: str  # which LLM produced this
    turn_id: Optional[str] = None  # transcript id
