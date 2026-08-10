# HR Policy MCP Server

This directory implements an MCP (Model Context Protocol) server, built with
the official `mcp` Python SDK's `FastMCP` high-level API, exposed over
**stdio transport**. It is launched as a local subprocess by the agent
orchestrator's MCP client (`app/agent/mcp_client.py`) — see
`design-and-evaluation.md` for the full architecture diagram and rationale.

Run it standalone for manual testing / debugging:

```bash
python -m mcp_server.server
```

It will sit waiting for MCP JSON-RPC messages on stdin/stdout; use an MCP
client (or `scripts/demo_pto.py` / the test suite) to talk to it rather than
typing at it directly.

## Tools

Each tool below is a real MCP tool (verified via a live `tools/list` call
against the running server — see `tests/test_mcp_tools.py`). Two tools
(`search_policy_documents`, `check_policy_compliance`) are backed by the RAG
index (`app/rag/retrieval.py`); four tools (`lookup_employee_profile`,
`check_pto_balance`, `lookup_benefits_status`, `create_mock_hr_ticket`) are
backed by the mock structured JSON data in `mock_data/`. `get_policy_section`
also uses the RAG index (direct metadata lookup rather than similarity
search). `draft_hr_email` is a pure template-based mock action with no
backing store.

### 1. `search_policy_documents`
RAG semantic search over the policy corpus. Returns top-k chunks + citations.

**Input schema**
```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "k": {"type": "integer", "default": 4}
  },
  "required": ["query"]
}
```

### 2. `get_policy_section`
Fetch chunk(s) for a specific policy document by `doc_id`, optionally
filtered to a section heading (case-insensitive substring match).

**Input schema**
```json
{
  "type": "object",
  "properties": {
    "doc_id": {"type": "string"},
    "section": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}
  },
  "required": ["doc_id"]
}
```

### 3. `check_policy_compliance`
RAG-grounded compliance judgment for a described scenario: retrieves policy
evidence, applies a transparent keyword-flag heuristic (`requires_approval`,
`possible_ineligibility`, `exception_process_may_apply`,
`possible_policy_violation`) over the retrieved text, and returns a judgment
label plus the underlying evidence chunks and citations. When the
orchestrator's LLM layer is available (`ANTHROPIC_API_KEY` set), it uses this
same evidence to produce a fuller natural-language explanation; the tool
itself never requires an LLM call.

**Input schema**
```json
{
  "type": "object",
  "properties": {
    "scenario": {"type": "string"},
    "employee_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}
  },
  "required": ["scenario"]
}
```

### 4. `lookup_employee_profile`
Mock employee directory lookup (name, role, department, manager_id,
location/state/country, employment_type, hire_date) by `employee_id`.

**Input schema**
```json
{"type": "object", "properties": {"employee_id": {"type": "string"}}, "required": ["employee_id"]}
```

### 5. `check_pto_balance`
Mock PTO balance + accrual-rate lookup by `employee_id`.

**Input schema**
```json
{"type": "object", "properties": {"employee_id": {"type": "string"}}, "required": ["employee_id"]}
```

### 6. `lookup_benefits_status`
Mock benefits elections/eligibility lookup by `employee_id`.

**Input schema**
```json
{"type": "object", "properties": {"employee_id": {"type": "string"}}, "required": ["employee_id"]}
```

### 7. `create_mock_hr_ticket`
**Mock, write action — requires explicit confirmation.** Appends a ticket to
`mock_data/tickets.json`. No real ticketing system is ever contacted. If
`confirm` is not `true`, nothing is written; the tool instead returns a
preview of what would be created (`needs_confirmation: true`) so the caller
can show the user the pending action and obtain explicit confirmation before
retrying with `confirm: true`. Every actual write is logged as
`MOCK ACTION: created mock HR ticket ...`.

**Input schema**
```json
{
  "type": "object",
  "properties": {
    "employee_id": {"type": "string"},
    "subject": {"type": "string"},
    "description": {"type": "string"},
    "category": {"type": "string", "default": "general"},
    "confirm": {"type": "boolean", "default": false}
  },
  "required": ["employee_id", "subject", "description"]
}
```

### 8. `draft_hr_email`
**Mock action — never sends anything.** Returns a templated draft
(`to_employee_id`, `subject`, `body`) composed from the provided key points.
No mail system is contacted; this is a pure, side-effect-free text generator.

**Input schema**
```json
{
  "type": "object",
  "properties": {
    "to_employee_id": {"type": "string"},
    "subject": {"type": "string"},
    "key_points": {"type": "string"},
    "from_name": {"type": "string", "default": "HR Assistant"}
  },
  "required": ["to_employee_id", "subject", "key_points"]
}
```

## Safety notes

- The only tool with real (mock) side effects is `create_mock_hr_ticket`,
  and it is gated behind `confirm=true`. The orchestrator (see
  `app/agent/orchestrator.py`) never sets `confirm=true` on the first pass —
  it always surfaces the pending action to the user first and only calls the
  tool again with `confirm=true` after the user's next message confirms.
- `draft_hr_email` never actually sends email; it is included to demonstrate
  a reversible/no-op mock action pattern distinct from the irreversible
  ticket-creation action.
- All employee data in `mock_data/` is synthetic (see `mock_data/employees.json`
  etc.) and no tool here ever reads or writes real personal data.
