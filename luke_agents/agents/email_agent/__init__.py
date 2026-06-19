"""LukeMail email-template-builder agent.

Chat-to-build-emails: the user describes an email in natural language; the LLM
returns the FULL bounded EmailDoc each turn; Python repairs/clamps it so the UI
(luke-consumer-ui) always gets a valid document to render and store as the
editable source (Postmark owns the rendered HTML). The same agent also generates
sample merge-variable values (LukeTests) for the builder's live preview + test send.
"""
from .agent import EmailAgent

__all__ = ["EmailAgent"]
