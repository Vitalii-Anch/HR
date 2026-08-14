# Deployed Instance

**Status:** Live on Render (free tier).

- **Deployed URL:** https://hr-agentic-rag-pdxv.onrender.com
- **Health check URL:** https://hr-agentic-rag-pdxv.onrender.com/health
- **Deploy date:** 2026-08-12
- **Render service name:** hr-agentic-rag
- **Anthropic API key configured:** yes (set as a Render dashboard secret env
  var per `render.yaml`'s `sync: false`; never committed to the repo)
- **Notes on first real-key run:** Verified locally first (see
  `ai-tooling.md`), then confirmed again against this deployed instance: the
  real Claude tool-use loop (`llm_used: true`) correctly runs both required
  demo workflows end-to-end, including the confirm-gated mock ticket
  creation flow. Two real issues were found and fixed while verifying
  against a real key that never surfaced in the mocked-client test suite:
  (1) `/chat` is stateless across HTTP calls, so the confirm-then-execute
  ticket flow needed an in-process pending-action store
  (`Orchestrator._execute_pending_action`) rather than relying on
  conversation memory that doesn't exist; (2) MCP tool calls had no timeout,
  so an intermittent stall in the local MCP subprocess could hang a request
  indefinitely -- fixed with a 20s timeout wrapper (`Orchestrator._call_tool`)
  around every tool call, which was observed in practice to recover
  gracefully from a real transient stall during evaluation. See
  `design-and-evaluation.md` and `evaluation/results.md` for full detail.

## Cold-start behavior

Render's free tier spins the service down after a period of inactivity. The
first request after a spin-down triggers Render's own "waking up" splash
page (visible in-browser) while Render brings up a fresh container and the
app's own startup sequence runs (connecting the MCP client to the MCP
server subprocess over stdio; the Chroma index and TF-IDF vocabulary are
pre-built at deploy time, not at request time). This typically takes well
under Render's own spin-up window; if demoing live, hit `/health` once
before the recorded portion begins just to confirm the instance is awake.

## Resolved: OOM crash on the first RAG-backed call

**Root cause found and fixed.** Earlier deploys crashed with "Ran out of
memory (used over 512MB)" and/or a health-check timeout, reliably on the
first `search_policy_documents`/`check_policy_compliance` call in a fresh
instance. Root cause: importing PyTorch (via `sentence-transformers`) inside
the MCP server subprocess to serve that first call added 200-400MB of
resident memory on top of the app's baseline, which exceeded the free
tier's 512MB container cap.

**Fix:** replaced the PyTorch/sentence-transformers embedding pipeline with
a pure-Python TF-IDF implementation (`app/rag/embeddings.py`) -- no PyTorch,
no model weights, negligible memory. See `design-and-evaluation.md` §2.2
for the full design writeup and retrieval-quality tradeoff discussion, and
`evaluation/results.md` for updated metrics.

**Measured impact:** running both demo workflows (PTO + remote-work
eligibility) back to back against a fresh local instance, combined resident
memory across the main process and the MCP subprocess was ~229MB (main
~119MB + MCP subprocess ~110MB) -- well within the 512MB cap, versus
reliably exceeding it before this fix. Also fixed in the same pass: the MCP
subprocess wasn't inheriting this project's environment variables at all
(a separate pre-existing bug in `app/agent/mcp_client.py` -- the MCP SDK's
`stdio_client` only passes through a small hardcoded safe-list of env vars
unless told otherwise), so it was silently running on `app/config.py`'s
hardcoded defaults rather than whatever `render.yaml`/the Render dashboard
actually configured. Those defaults happened to match render.yaml's values,
so this didn't cause a visible bug, but it's now fixed for correctness.

## Post-deploy smoke test checklist

Confirmed against the live URL above (see `scripts/demo_pto.ps1` /
`scripts/demo_remote.ps1`, or the original `.sh` equivalents with
`BASE_URL=https://hr-agentic-rag-pdxv.onrender.com`):

- [x] `GET /health` returns `{"status": "ok", "mcp_connected": true, "index_loaded": true}`
- [ ] `GET /` renders the chat UI
- [ ] A plain policy question (e.g. "What is the PTO carryover limit?")
      returns an answer with citations
- [ ] The PTO demo workflow (`scripts/demo_pto.ps1`) completes, including the
      confirm-gated mock ticket creation
- [ ] The remote-work demo workflow (`scripts/demo_remote.ps1`) completes,
      including the clarifying-question and out-of-scope-refusal steps
- [x] With `ANTHROPIC_API_KEY` set, a response shows `"llm_used": true` and
      real Claude-generated prose instead of the fallback template text
