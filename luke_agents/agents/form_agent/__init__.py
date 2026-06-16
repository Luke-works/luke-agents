"""LukeBuilds form-builder agent (with LukeTests data generation).

Chat-to-build-forms: the user describes a form in natural language; the LLM
returns targeted operations plus a conversational reply each turn; Python applies
them and renders the coltorapps builder schema that luke-consumer-ui /
luke-capability-engine consume. The same agent also generates test data
(LukeTests) for the builder's Test feature.
"""
from .agent import FormAgent

__all__ = ["FormAgent"]
