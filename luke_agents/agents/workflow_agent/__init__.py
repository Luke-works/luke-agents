"""LukeFlow workflow-builder agent.

Chat-to-build-workflows: the user describes an automation in natural language; the
LLM returns the FULL bounded WorkflowDoc each turn; Python repairs dangling
references so the visual builder (luke-consumer-ui) always gets a valid, wireable
graph to render and store as the editable draft (luke-core-engine compiles it to
BPMN). Mirrors the email + form agents behind the shared `Agent` contract.
"""
from .agent import WorkflowAgent

__all__ = ["WorkflowAgent"]
