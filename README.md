# luke-agents

A small platform for hosting many LLM agents behind **one** FastAPI app and one
(cheap) deployment. Each agent is a self-contained package; the shared `core`
layer handles the parts every agent needs — LLM brain selection, per-caller rate
limiting, and the server that mounts agents under `/agents/<slug>`.

The first agent is **`form_agent`**: chat-to-build-forms (**LukeBuilds**), emitting
the coltorapps builder schema that luke-consumer-ui / luke-capability-engine
consume, plus test-data generation (**LukeTests**) for the builder's Test feature.
It was extracted from the standalone `luke-form-agent` service.

## Layout

```
luke-agents/
  main.py                          # uvicorn entrypoint: registers agents -> build_app(...)
  luke_agents/
    core/                          # agent-agnostic plumbing
      llm.py                       #   brain selection (Groq | OpenAI gpt-5-nano | Gemini | Ollama) + typed generate()
      ratelimit.py                 #   per-caller sliding-window limiter (+ enforce() -> HTTP 429)
      registry.py                  #   the Agent contract (AgentMeta + Agent base class)
      server.py                    #   build_app(): CORS, /health, mounting, root landing
    agents/
      form_agent/                  # one agent = one package
        agent.py                   #   FormAgent: the /chat route, wired to core
        prompt.py                  #   LukeBuilds + LukeTests system prompts + builders
        schema.py                  #   FormSpec / AssistantTurn / Chat request+response
        coltorapps.py              #   FormSpec <-> coltorapps builder schema
        static/index.html          #   browser test client
    core/transcripts.py            # (shared) record turns -> Postgres | JSONL
    tools/export_finetune.py       # turn transcripts into fine-tuning JSONL
```

## Routes

- `GET  /health` — active brain + list of mounted agents
- `GET  /` — the default agent's UI (or a landing page listing agents)
- `POST /agents/form/chat` — the form agent
- `GET  /agents/form/` — the form agent's test client
- `POST /chat` — alias for the **default** agent (drop-in for existing
  single-agent clients such as luke-consumer-ui's `VITE_FORM_AGENT_URL`)

## Adding an agent

1. Create `luke_agents/agents/<your_agent>/` with an `Agent` subclass that sets
   `meta` and implements `build_router()` (and optionally `static_index()`).
2. Use `core.llm.generate(system, user, ResponseModel)` for the LLM call and
   `core.ratelimit.enforce(key)` to bound spend.
3. Register it in `main.py`'s `AGENTS` list. Done — it mounts at `/agents/<slug>`.

## Retaining chats for fine-tuning

Every `/chat` turn is recorded as a ready-made supervised example — the exact
`messages` sent (system + user) and the model's JSON `output` — plus metadata and
a quality signal. `core/transcripts.py` is the shared (agent-agnostic) store:

- **`DATABASE_URL` set** → **Postgres**. Tables are created in the
  **`luke_agents`** schema (`AGENTS_DB_SCHEMA`) of the shared luke Postgres —
  durable, survives Render redeploys.
- **unset** → append-only **JSONL** under `TRANSCRIPTS_DIR` (local dev only;
  Render's disk is ephemeral).
- **`TRANSCRIPTS_ENABLED=false`** → record nothing.

Recording happens in a background task (zero added latency) and can never fail a
chat request. Each response includes a `turn_id`.

**Quality labels** (so you train on good turns only):
- *Automatic*: the `changed` flag, errors, and latency are captured per turn.
- *Explicit*: `POST /agents/form/feedback {turn_id, accepted, rating, note}` lets
  the UI mark a turn kept/undone or 👍/👎. `accepted:false` and `rating:-1` are
  excluded from exports.

**Privacy**: form content can contain PII. Pass `consent:false` in a `/chat` body
to exclude that turn from training; the exporter drops non-consented turns. Use a
pseudonymous `user_id`, and tell users chats may be retained for model improvement.

**Export to fine-tuning JSONL** (provider-neutral chat format):

```bash
# default: kept/changed turns, consented, no 👎
python -m luke_agents.tools.export_finetune --agent form --out form_sft.jsonl
# stricter: only turns a user explicitly kept
python -m luke_agents.tools.export_finetune --only-accepted --out form_sft.jsonl
```

Output is `{"messages":[{system},{user},{assistant}]}` per line — directly usable
for an OpenAI/Gemini fine-tune or to SFT an open model (Llama/Qwen). Note: Groq
*serves* open models but doesn't host fine-tuning of arbitrary models, so the
fine-tune itself runs on a provider/toolchain that does; this format converts
cleanly to each.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paste a Groq key, OR leave blank + run Ollama
uvicorn main:app --reload
# open http://localhost:8000
```

Get a free Groq key (email signup, no card): https://console.groq.com/keys

## Deploy to Render

1. Push this folder to a GitHub repo.
2. In Render: **New ➜ Blueprint** (picks up `render.yaml`).
3. Add `GROQ_API_KEY` in the dashboard, deploy, visit the service URL.

> Migrating off the old `luke-form-agent` deployment: this app serves the form
> agent at the root `POST /chat` too, so pointing `VITE_FORM_AGENT_URL` at the
> new service URL is a drop-in swap. New clients should prefer
> `/agents/form/chat`.
