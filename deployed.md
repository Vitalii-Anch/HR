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
page (visible in-browser), followed by the app's own startup sequence
(loading the local embedding model, rebuilding the Chroma index from
`corpus/`, and connecting the MCP client to the MCP server subprocess over
stdio). In practice this cold start takes roughly 30-90 seconds end to end;
subsequent requests are fast (warm). If demoing this live, expect a delay on
the very first request and plan the recording around it (e.g. hit `/health`
once before the recorded portion begins to warm the instance up).

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
