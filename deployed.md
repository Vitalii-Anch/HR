# Deployed Instance

**Status:** Not yet deployed. This file is a placeholder to be filled in
after a real deployment to Render (see `render.yaml` and the "Deployment"
section of `README.md`).

## To fill in after deploying

- **Deployed URL:** `https://<service-name>.onrender.com` — TBD
- **Health check URL:** `https://<service-name>.onrender.com/health` — TBD
- **Deploy date:** TBD
- **Render service name:** TBD
- **Anthropic API key configured:** yes / no (do not paste the key here)
- **Notes on first real-key run:** TBD (e.g. any differences observed
  between the deterministic fallback mode used during development and the
  real Claude tool-use loop, per `ai-tooling.md`'s noted verification gap)

## Post-deploy smoke test checklist

Once deployed, confirm the following against the live URL (see
`scripts/demo_pto.sh` / `scripts/demo_remote.sh`, run with
`BASE_URL=https://<service-name>.onrender.com`):

- [ ] `GET /health` returns `{"status": "ok", "mcp_connected": true, "index_loaded": true}`
- [ ] `GET /` renders the chat UI
- [ ] A plain policy question (e.g. "What is the PTO carryover limit?")
      returns an answer with citations
- [ ] The PTO demo workflow (`scripts/demo_pto.sh`) completes, including the
      confirm-gated mock ticket creation
- [ ] The remote-work demo workflow (`scripts/demo_remote.sh`) completes,
      including the clarifying-question and out-of-scope-refusal steps
- [ ] With `ANTHROPIC_API_KEY` set, a response shows `"llm_used": true` and
      real Claude-generated prose instead of the fallback template text
