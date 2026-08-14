# Northwind HR Agentic RAG

An agentic AI system that helps employees of a fictional company ("Northwind
Retail Co.") complete HR policy and operations tasks: answering policy
questions with citations, checking PTO balances and benefits, evaluating
remote-work eligibility, and (with explicit confirmation) filing mock HR
tickets or drafting mock HR emails. Built as a graduate AI Engineering course
project (Quantic "AI Engineering Techniques and Architectures").

See **`design-and-evaluation.md`** for the full architecture, design
rationale, and description of the two required demo workflows;
**`ai-tooling.md`** for how this project was built; **`deployed.md`** for the
deployment URL (filled in after a real deploy); **`mcp_server/README.md`**
for the MCP tool catalog.

## Architecture at a glance

```
Browser --HTTP--> FastAPI app (app/main.py) --> Agent Orchestrator (app/agent/)
                                                     |
                                     MCP Client (stdio) <--protocol--> MCP Server subprocess (mcp_server/)
                                                     |                        |
                                              Claude (Anthropic API)     RAG index (Chroma) + mock_data/*.json
```

One deployed service. The MCP server runs as a local subprocess of the same
process that serves the web app (no second deployed service, no network hop).
See `design-and-evaluation.md` for the full diagram.

## Requirements

- Python 3.10
- An `ANTHROPIC_API_KEY` **only if** you want real LLM-synthesized answers.
  Everything else (RAG retrieval, MCP tools, guardrails, both demo
  workflows) works without one, via a deterministic fallback path — this is
  intentional (see "Running without an API key" below).

## Setup (local development)

```bash
cd hr-agentic-rag
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# On some minimal/sandboxed Linux environments you may need:
#   pip install --break-system-packages -r requirements.txt

cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY if you have one. Everything works
# without it (see below).
```

### Embedding model

This project embeds policy chunks with a lightweight, pure-Python TF-IDF
implementation (`app/rag/embeddings.py`) — no PyTorch, no ONNX runtime, no
model weights to download, and no network dependency of any kind. This
replaced an earlier `sentence-transformers/all-MiniLM-L6-v2` implementation,
which worked but imported PyTorch inside the MCP server subprocess, adding
200-400MB of resident memory that reliably exceeded Render free tier's
512MB container cap (see `deployed.md` and `design-and-evaluation.md` §2.2
for the full incident and tradeoff writeup).

### Build the RAG index

```bash
python scripts/build_index.py
```

This is idempotent — re-run any time you edit `corpus/`. It also runs
automatically on app startup if the Chroma collection is empty.

### Run the app

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 for the chat UI, or:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the PTO carryover limit?"}'
```

Try employee id `E1002` (Software Engineer, remote) for personalized
questions like "What's my PTO balance?".

### Reproduce the two required demo workflows without the UI

```bash
# In one terminal:
uvicorn app.main:app --host 0.0.0.0 --port 8000
# In another terminal:
bash scripts/demo_pto.sh       # (a) PTO request guidance workflow
bash scripts/demo_remote.sh    # (b) Remote work eligibility workflow
```

Both scripts are plain `curl` calls against a running server and print the
full JSON response (answer, citations, tool_trace) at each step, including
the confirm-then-execute flow for mock ticket creation. See
`design-and-evaluation.md` for the expected tool-call sequence each workflow
should produce.

## Running without an API key

This project was built and CI-tested in a sandbox with **no
`ANTHROPIC_API_KEY` available**, by design requirement. The orchestrator
(`app/agent/orchestrator.py`) checks whether a key is configured at request
time:

- **Key set:** runs a real Claude tool-use agentic loop (the "real" mode).
- **Key not set:** runs a deterministic, template-based fallback that still
  exercises the real MCP tools and real RAG retrieval — every tool call and
  guardrail is genuine, only the final answer's *prose* is templated instead
  of LLM-generated. Every such response is labeled `llm_used: false` with an
  explicit note in the answer text.

This means the full test suite, the demo workflows, and most of the
evaluation harness all run and pass with zero API cost and no key. Once you
add a real key to `.env` (or the deployment environment), the exact same
code paths automatically switch to real LLM synthesis — no code changes
required.

## Tests

```bash
pytest -v
```

23 tests, all passing without any API key: an app import/startup smoke test,
real MCP tool discovery + tool-call tests (over the actual stdio protocol,
not direct function calls), RAG retrieval tests, and orchestrator guardrail
tests — including LLM-loop tests where the Anthropic call is fully
mocked/stubbed (see `tests/test_orchestrator.py::StubLLMClient`).

## Evaluation

```bash
python evaluation/run_eval.py
```

Runs the 30-item `evaluation/eval_set.json` through the full pipeline and a
k=2/4/6 retrieval-only ablation, writing `evaluation/results.json`. See
`evaluation/results.md` for the real numbers from this project's build
environment (retrieval metrics are final; LLM-dependent answer quality will
improve once a real key is supplied, without any code changes).

## Deployment

Target: [Render](https://render.com) free tier, single web service (see
`render.yaml`). Fill in `deployed.md` with the live URL once deployed. The
Chroma index and embedding model weights are both rebuilt/re-extracted at
build time (see `render.yaml`'s `buildCommand`), so no persistent disk is
required — appropriate for Render's free tier, which doesn't offer one.

## Directory layout

```
app/                  FastAPI app, agent orchestrator, RAG pipeline
  agent/               MCP client, LLM client, orchestrator loop, trace types
  rag/                 chunking, parsing, embeddings, ingestion, retrieval
  web/templates/       minimal server-rendered chat UI
mcp_server/            MCP server (8 tools) + README documenting tool schemas
corpus/                12 synthetic HR policy documents (.md / .html / .txt)
mock_data/             synthetic employees, PTO balances, benefits, tickets
scripts/               build_index.py, demo_pto.sh, demo_remote.sh
tests/                 pytest suite (no API key required)
evaluation/            eval_set.json, run_eval.py, results.md/.json
design-and-evaluation.md   architecture, design rationale, demo workflows
ai-tooling.md              how AI tooling was used to build this project
deployed.md                deployment URL placeholder (filled in later)
```

## Known gaps / what still needs a real API key or real deployment

- Real LLM-synthesized answers and the LLM-backed agentic loop have only
  been verified with a fully mocked/stubbed Anthropic client (see
  `tests/test_orchestrator.py`), never against the live API, since no key
  was available in this build environment.
- `evaluation/results.md`'s full-pipeline metrics reflect the deterministic
  fallback mode; re-run `evaluation/run_eval.py` with a real key for final
  LLM-mode numbers.
- `deployed.md` is a placeholder until an actual Render deployment is done.
