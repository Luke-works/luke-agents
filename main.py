"""ASGI entrypoint:  uvicorn main:app

Register agents here. Each is a self-contained package under
`luke_agents/agents/`; `build_app` mounts every one under `/agents/<slug>` and
exposes the chosen default at the root too (drop-in for single-agent clients).
"""
from dotenv import load_dotenv

from luke_agents.agents.form_agent import FormAgent
from luke_agents.core import build_app

load_dotenv()

AGENTS = [
    FormAgent(),
    # Add more agents here, e.g. WorkflowAgent(), EmailAgent(), ...
]

app = build_app(AGENTS, default_slug="form")
