# Design and Evaluation

## 1. Architecture

Single deployed service (fits Render's free tier: one web process, no
external database, no second deployed service for the MCP server).

```
                              ┌───────────────────────────────────────────────────────────┐
                              │                     Render web service                     │
                              │                                                             │
  Browser / curl              │  FastAPI app (app/main.py)                                 │
  ─────────HTTP──────────────▶│    GET  /            -> minimal server-rendered chat UI    │
  GET /health                 │    GET  /health      -> {status, mcp_connected, index_loaded}
  POST /chat                  │    POST /chat        -> ChatRequest -> Orchestrator        │
                              │             │                                               │
                              │             ▼                                               │
                              │  Agent Orchestrator (app/agent/orchestrator.py)             │
                              │    - guardrails (clarify / escalate / confirm-gate)         │
                              │    - LLM mode (if ANTHROPIC_API_KEY set) OR                 │
                              │      deterministic fallback mode (no key)                   │
                              │             │                              │                │
                              │             ▼                              ▼                │
                              │  LLM Client (app/agent/llm_client.py)   MCP Client            │
                              │    - anthropic SDK, claude-sonnet-5     (app/agent/mcp_client.py)
                              │    - reads ANTHROPIC_API_KEY lazily       - stdio subprocess  │
                              │             │                              │                │
                              │             ▼                              ▼                │
                              │      Anthropic API (network)      MCP Server subprocess      │
                              │      (only when key is set)       (mcp_server/server.py)      │
                              │                                    - FastMCP, stdio transport │
                              │                                    - 8 tools (see below)      │
                              │                                         │           │         │
                              │                                         ▼           ▼         │
                              │                                  RAG index      mock_data/     │
                              │                                  (Chroma,       *.json          │
                              │                                  local          (employees,     │
                              │                                  sentence-      pto_balances,   │
                              │                                  transformers   benefits,       │
                              │                                  embeddings)    tickets)        │
                              └───────────────────────────────────────────────────────────┘
```

Key property: the MCP server is a **local subprocess** of the same process
serving the web app, launched over **stdio transport** by the MCP client.
There is no second deployed service, no inter-service network hop, and no
inter-service auth to manage — appropriate for a free-tier, single-service
deployment. The orchestrator never imports `mcp_server` functions directly;
every tool invocation is a real MCP `tools/list` / `tools/call` round trip
(verified in `tests/test_mcp_tools.py`).

## 2. Design decisions and rationale

### 2.1 Chunking strategy (heading-aware, with overlap)

Implemented in `app/rag/chunking.py`. Corpus documents are first split along
their own heading structure (`## ` in Markdown, `<h2>` in HTML, `SECTION N:`
in the plain-text document) so each chunk maps to one coherent policy
section (e.g. "PTO Policy > Blackout Periods") rather than an arbitrary byte
window. Any section longer than 800 characters is further split into
overlapping 800-character windows with 150 characters of overlap, so a
sentence spanning a chunk boundary still appears in full in at least one
chunk. Because most sections in this corpus are already under 800
characters, the overlapping-window logic mainly protects the handful of
longer sections (e.g. Remote Work Policy > Tax and Legal Considerations);
most chunks in the index *are* whole sections. This keeps citations
meaningful (a citation is "doc_id + section heading", not an offset) and
avoids diluting embedding signal by cramming multiple unrelated topics into
one chunk.

### 2.2 Embedding model choice

**Current implementation: pure-Python TF-IDF** (`app/rag/embeddings.py`), no
PyTorch, no ONNX runtime, no model weights. This replaced an earlier
implementation that used `sentence-transformers/all-MiniLM-L6-v2` (a small,
~90MB, locally-run neural embedding model). That implementation worked
correctly and produced good retrieval quality in local testing and CI, but
importing PyTorch inside the MCP server subprocess reliably added 200-400MB
of resident memory on top of the rest of the app's footprint, which
exceeded Render free tier's 512MB container memory cap the first time a
RAG-backed tool call loaded it — a reproducible production OOM crash (see
`deployed.md` for the incident and the memory measurements after the fix).

Given a small (12-document, ~56KB), topically well-separated HR policy
corpus, and a hard free-tier memory budget, TF-IDF cosine similarity is a
well-understood classical-IR alternative to dense neural embeddings: no
large model weights, negligible memory (a fitted vocabulary + IDF table,
serialized as one ~45KB JSON file), and effectively instant to compute (no
model load time at all, versus a multi-second PyTorch import). Retrieval
still needs no API key or network dependency, satisfying the same hard
requirement that the retrieval half of the system work without
`ANTHROPIC_API_KEY`.

**Tradeoff:** TF-IDF matches on literal/lexical term overlap rather than
learned semantic similarity, so a paraphrased query that shares little
vocabulary with the relevant chunk retrieves worse than with dense neural
embeddings would. Two mitigations are implemented: (1) each chunk's
embedding is computed from its document title + section heading + body
text, not body text alone, so a query term that only appears in a heading
(e.g. "carryover" in "Maximum Balance and Carryover", where the body always
writes "carry over" as two words) still matches; (2) a hand-rolled English
stopword list is filtered out of both documents and queries before
vectorizing, since without it, generic connective words shared by nearly
every chunk (including boilerplate "Purpose"/"Overview" sections) dominate
the similarity score and regularly outrank the chunk that actually contains
the query's distinguishing term. See `evaluation/results.md` for retrieval
metrics measured under this design (recall@4 ≈ 0.90 on the eval set).

### 2.3 Vector store: Chroma (persistent, local)

No server process to run, no cloud account, persists to a plain on-disk
directory (`CHROMA_PERSIST_DIR`), and its Python API stores arbitrary
per-chunk metadata alongside vectors, which citations depend on (doc_id,
title, section heading, source format, source path). This fits the
single-service, free-tier deployment constraint directly.

*Build-sandbox note:* this project's sandbox mounted its output directory
over FUSE, and Chroma's SQLite-backed storage engine cannot acquire the file
locks it needs on that mount (`disk I/O error`). This is purely an artifact
of that sandbox's filesystem, not of the code or of Render/any normal
filesystem — verification in this sandbox pointed `CHROMA_PERSIST_DIR` at a
local ext4 scratch path via a (gitignored) `.env` override; the default
`./data/chroma` is what ships and is what a normal filesystem (a developer's
machine, GitHub Actions runners, Render) uses without issue.

### 2.4 Retrieval k and keyword-boost rerank

Default `RETRIEVAL_TOP_K=4` (configurable via env var), chosen from the k=2/4/6
ablation in `evaluation/results.md`: recall@k is already 1.0 at k=2 for this
corpus, but **mean gold-document coverage for multi-document questions**
rises from 0.90 (k=2) to 0.95 (k=4) to 0.98 (k=6). k=4 captures most of the
multi-document benefit of k=6 while returning 33% fewer chunks (less noise
for LLM synthesis, smaller prompt) with no measurable latency cost
difference between k=4 and k=6 once the model is warm.

`app/rag/retrieval.py` also applies a small keyword-overlap boost
(`keyword_boost_weight=0.15`) on top of embedding similarity: a chunk that
literally contains the user's query terms gets nudged slightly higher,
without overriding semantic similarity. This is a cheap, dependency-free way
to correct the occasional case where dense similarity ranks a topically
related but less specific chunk above one containing an exact term the user
asked about.

### 2.5 MCP transport: stdio

The `mcp` Python SDK supports stdio, SSE, and streamable-HTTP transports.
stdio was chosen because the MCP server and its client live in the same
deployed process/host (see Section 1) — there is no reason to add a network
transport (and its associated latency, serialization overhead, and
auth/CORS concerns) when a subprocess pipe is simpler, faster, and
sufficient. This also matches the free-tier "single deployed service"
constraint directly: SSE/HTTP transport would imply a second listening
port/service to manage.

### 2.6 MCP tool schemas

Full JSON schemas (as actually reported by a live `tools/list` call) are
documented in `mcp_server/README.md`. Summary:

| Tool | Backed by | Notes |
|---|---|---|
| `search_policy_documents` | RAG | top-k semantic search + citations |
| `get_policy_section` | RAG | exact doc_id (+ optional section) lookup |
| `check_policy_compliance` | RAG | scenario -> evidence + heuristic judgment + citations |
| `lookup_employee_profile` | mock_data | employee directory lookup |
| `check_pto_balance` | mock_data | PTO balance + accrual rate |
| `lookup_benefits_status` | mock_data | benefits elections/eligibility |
| `create_mock_hr_ticket` | mock_data | **write, gated on `confirm=true`** |
| `draft_hr_email` | template | mock draft only, never sends |

### 2.7 Safety guardrails

Implemented in `app/agent/orchestrator.py`, enforced in code (not only via
LLM system-prompt instructions, so they hold even in fallback mode):

1. **Clarifying questions:** if a message implies a specific employee's
   data ("my PTO", "my balance", "am I eligible", etc.) but no `employee_id`
   was supplied, the orchestrator asks for it before calling any tool.
2. **Out-of-corpus refusal:** if the top retrieval score for a query is
   below `OUT_OF_SCOPE_SIMILARITY_THRESHOLD` (0.28), the orchestrator
   declines and suggests contacting HR directly, rather than answering from
   weak or irrelevant evidence.
3. **Confirm-gated write action:** `create_mock_hr_ticket` only ever
   executes with `confirm=true` if the *human* set `confirm=true` on the
   `/chat` request. Even in LLM mode, if Claude's tool-use call sets
   `confirm=true` on its own, the orchestrator **overwrites** it with the
   human-supplied value before executing (`_safe_tool_arguments`); this is
   tested explicitly in
   `tests/test_orchestrator.py::test_llm_loop_never_trusts_llm_confirm_flag`.
   Calling the tool without confirmation returns a preview
   (`needs_confirmation: true`) with nothing written.
4. **`draft_hr_email` is a true no-op:** it never contacts a mail system; it
   is included specifically to demonstrate the reversible/no-op case,
   distinct from the irreversible, confirm-gated ticket-creation action.
5. **Graceful failure handling:** an unknown `employee_id` (e.g.
   `check_pto_balance` returning `found: false`) produces an escalation
   message rather than a fabricated balance; an LLM call failure (e.g. rate
   limit, malformed key) is caught and the orchestrator falls back to the
   deterministic path with a clear note rather than crashing or silently
   returning nothing.

### 2.8 Deployment architecture

Render free tier, single web service (`render.yaml`): `buildCommand`
installs dependencies and runs `scripts/build_index.py` to (re)build the
Chroma index from `corpus/`; `startCommand` runs `uvicorn app.main:app`.
`ANTHROPIC_API_KEY` is a dashboard secret (`sync: false`), never committed.
No persistent disk is configured (Render's free tier doesn't offer one, and
none is needed): both the Chroma index and the local embedding model weights
are fully rebuilt/re-extracted on every deploy, so there is no dependency on
state surviving a redeploy or a free-tier spin-down/spin-up cycle.

## 3. The two required demo workflows

### (a) PTO request guidance

**Trigger:** a message mentioning PTO/vacation/time off, with an
`employee_id`.

**Expected MCP tool-call sequence:**
1. `check_pto_balance(employee_id)` — get the employee's current balance and
   accrual rate.
2. `search_policy_documents("PTO accrual balance manager approval requesting time off")`
   — retrieve the manager-approval and advance-notice requirements from the
   PTO Policy.
3. *(only if the user asks to submit a request, and only ever previewed
   first)* `create_mock_hr_ticket(..., confirm=false)` — returns a preview,
   `needs_confirmation: true`, nothing written.
4. *(only after the human explicitly sets `confirm=true` on a follow-up
   `/chat` call)* `create_mock_hr_ticket(..., confirm=true)` — actually
   creates the mock ticket.

**Reproduce:** `bash scripts/demo_pto.sh` (two `/chat` calls: one without
confirmation, one with).

### (b) Remote work eligibility

**Trigger:** a message asking about remote work eligibility, with an
`employee_id`.

**Expected MCP tool-call sequence:**
1. `lookup_employee_profile(employee_id)` — get role, department, current
   location, employment type (role determines base remote eligibility per
   the Remote Work Policy).
2. `check_policy_compliance(scenario, employee_id)` — internally runs RAG
   retrieval over the Remote Work / Data Security policies and returns a
   judgment (`likely_compliant` / `compliant_with_conditions` /
   `likely_not_compliant_or_ineligible` / `insufficient_evidence`) with
   flags and citations.
3. Final answer cites the specific policy sections behind the judgment and
   states next steps (e.g. "requires manager + Payroll tax-registration
   approval for a different U.S. state" or "cross-border remote work
   requires Legal/People & Culture sign-off and is not automatically
   approved").

**Reproduce:** `bash scripts/demo_remote.sh` (also demonstrates the
clarifying-question guardrail when `employee_id` is omitted, and the
out-of-scope refusal for an unrelated question).

## 4. Evaluation summary

See `evaluation/results.md` for full numbers and interpretation; headline
results from this project's build environment (no API key):

- Retrieval-only ablation: recall@k = 1.0 at k=2/4/6; multi-document gold-doc
  coverage improves from 0.90 (k=2) to 0.95 (k=4) to 0.98 (k=6), justifying
  the default k=4.
- Full pipeline (fallback mode): groundedness_rate = 1.0, tool_selection_accuracy
  = 1.0, workflow_completion_rate = 1.0, escalation_clarification_accuracy =
  1.0, action_safety_pass_rate = 1.0, citation_recall_mean = 0.95, citation_precision_mean
  = 0.64 (multiple supporting citations are returned per answer by design).
- Latency in fallback mode: p50 ≈ 18ms, p95 ≈ 24ms (local-only; will be
  dominated by the Claude API round trip once a real key is configured).
