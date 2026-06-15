"""LukeTalks form-builder agent.

Chat-to-build-forms: the user describes a form in natural language; the LLM
returns the complete form (as a flat FormSpec) plus a conversational reply each
turn; Python renders that into the coltorapps builder schema that
luke-consumer-ui / luke-capability-engine consume.
"""
from .agent import FormAgent

__all__ = ["FormAgent"]
