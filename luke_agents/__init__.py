"""luke-agents — a small platform for hosting many LLM agents in one app.

`core` holds the reusable plumbing every agent shares (LLM brain selection,
rate limiting, the Agent contract, and the FastAPI server that mounts agents
under `/agents/<slug>`). `agents` holds the concrete agents; `form_agent`
(LukeTalks form builder) is the first one.
"""
