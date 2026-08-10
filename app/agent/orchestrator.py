"""
Agent orchestrator: the loop that takes a user message + minimal state,
decides whether RAG alone suffices or an MCP tool call is needed, executes
tool calls through the real MCP client (app/agent/mcp_client.py), and
synthesizes a final, cited answer.

Two operating modes:

1. LLM mode (ANTHROPIC_API_KEY set): a genuine agentic tool-use loop against
   Claude. Claude sees the MCP tools (converted to Anthropic's tool-use
   schema) and decides which to call; this module executes each call via the
   MCP client, feeds results back, and repeats until Claude produces a final
   text answer (or a turn limit is hit).

2. Deterministic fallback mode (no ANTHROPIC_API_KEY): a small keyword-based
   router still exercises the *same* MCP tools and RAG retrieval, and
   composes an answer by template rather than free-form generation. This
   exists so the two required demo workflows (PTO guidance, remote-work
   eligibility) are fully runnable and testable without any API key -- which
   is required for this project's CI and for the sandbox this project was
   built in. It is clearly labeled (`llm_used: False`) in every response so
   it is never mistaken for real LLM synthesis.

Guardrails implemented here (not inside the MCP tools, so they apply
regardless of mode):
  - Clarifying question when a request needs an employee_id but none was
    given (`_needs_employee_id`).
  - Out-of-corpus refusal when retrieval evidence is too weak
    (`OUT_OF_SCOPE_SIMILARITY_THRESHOLD`), instead of letting the model (or
    the fallback template) invent a policy answer.
  - Mock ticket creation is only ever executed with `confirm=True` if the
    *caller* (the human, via the `confirm` field on POST /chat) explicitly
    set it -- the LLM's own tool-call arguments are never trusted for this
    flag; see `_safe_tool_arguments`.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.agent.llm_client import LLMClient, LLMNotConfiguredError
from app.agent.mcp_client import MCPToolClient
from app.agent.trace import AgentResult, ToolCallTrace
from app.rag.retrieval import retrieve, format_citations

logger = logging.getLogger("hr_agentic_rag.orchestrator")

OUT_OF_SCOPE_SIMILARITY_THRESHOLD = 0.28
MAX_LLM_TURNS = 6
MCP_TOOL_CALL_TIMEOUT_SECONDS = 20.0

_NEEDS_EMPLOYEE_ID_PHRASES = [
    "my pto", "my balance", "my benefit", "my profile", "my remote",
    "check my", "am i eligible", "can i work remotely", "my vacation",
    "my time off", "my employee", "my hire date", "my manager",
]

SYSTEM_PROMPT = """You are the Northwind Retail Co. HR Assistant, an agentic AI that helps \
employees with HR policy questions and routine HR tasks using the provided tools.

Rules you MUST follow:
1. Ground every policy claim in retrieved evidence from `search_policy_documents`, \
`get_policy_section`, or `check_policy_compliance`. Always cite the doc_id and section \
you relied on in your final answer (e.g. "(PTO Policy > Eligibility and Accrual)").
2. If retrieval returns no sufficiently relevant evidence, say clearly that the \
question is outside the Northwind policy corpus and suggest contacting HR directly. \
Never invent a policy that wasn't retrieved.
3. Never answer a question about a *specific* employee's PTO balance, benefits, or \
profile unless you have an employee_id (from the tool results or the user's message). \
If you don't have one, ask for it.
4. `create_mock_hr_ticket` is a real (mock) write action, gated by the tool's own \
`confirm` argument -- you must ALWAYS call it with confirm=false first to get a safe, \
side-effect-free preview (never just describe the ticket in prose instead of calling \
the tool). As soon as you have enough information for subject/description/category \
(e.g. dates and reason are known), call `create_mock_hr_ticket(..., confirm=false)` \
immediately in that same turn -- do not ask "shall I go ahead?" in plain text without \
having called the tool, since the system can only remember and later execute a \
*previewed* action, not a described one. A "needs_confirmation" result is the expected, \
successful outcome of that preview call, not a failure -- summarize it for the user and \
tell them to confirm. Only skip calling the tool if a required detail (e.g. exact dates) \
is genuinely missing; in that case ask for it in text first, then call the tool once you \
have it.
5. `draft_hr_email` never sends anything; it only returns draft text. Present it as a draft.
6. Be concise, professional, and cite your sources.
"""


class Orchestrator:
    def __init__(self, mcp_client: MCPToolClient, llm_client: LLMClient | None = None):
        self.mcp_client = mcp_client
        self.llm_client = llm_client or LLMClient()
        # `/chat` is stateless -- each call is an independent request with no
        # conversation history. That's fine for one-shot Q&A, but the
        # confirm-then-execute ticket flow needs to remember *what* was
        # previewed so a later `confirm=true` call (which, on its own, has no
        # memory of the earlier turn's details) can execute the exact same
        # action rather than asking the user to repeat themselves or -- worse
        # -- having the LLM guess new ticket details from scratch. This is a
        # deliberately minimal in-process store (single free-tier service,
        # single worker process; not durable across restarts, which is fine
        # for this project's mock-action use case).
        self._pending_actions: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def handle_message(
        self,
        message: str,
        employee_id: str | None = None,
        confirm: bool = False,
    ) -> AgentResult:
        clarification = self._needs_employee_id(message, employee_id)
        if clarification:
            return AgentResult(
                answer=clarification,
                clarification_needed=True,
                basis="Guardrail: request implies a specific employee context but no employee_id was provided.",
            )

        if confirm and employee_id and employee_id in self._pending_actions:
            return await self._execute_pending_action(employee_id)

        if self.llm_client.is_configured():
            try:
                return await self._run_llm_loop(message, employee_id, confirm)
            except LLMNotConfiguredError as exc:
                logger.warning("LLM reported not configured mid-call: %s", exc)
                return await self._run_fallback(message, employee_id, confirm, llm_error=str(exc))
        else:
            return await self._run_fallback(message, employee_id, confirm)

    # ------------------------------------------------------------------
    # MCP tool calls, always timeout-guarded (see rubric requirement to
    # "handle failures gracefully, such as unavailable MCP tools"). Without
    # this, a stalled/crashed MCP subprocess hangs the whole request forever
    # with no way for the caller (or the demo!) to recover.
    # ------------------------------------------------------------------
    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            return await asyncio.wait_for(
                self.mcp_client.call_tool(name, arguments), timeout=MCP_TOOL_CALL_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error("MCP tool call timed out after %ss: %s(%r)", MCP_TOOL_CALL_TIMEOUT_SECONDS, name, arguments)
            return {
                "error": True,
                "detail": f"Tool '{name}' did not respond within {MCP_TOOL_CALL_TIMEOUT_SECONDS:.0f}s "
                          "(MCP server unresponsive). Nothing was written.",
            }
        except Exception as exc:  # noqa: BLE001 -- any transport/protocol failure should degrade, not hang
            logger.exception("MCP tool call failed: %s(%r)", name, arguments)
            return {"error": True, "detail": f"Tool '{name}' failed: {exc}"}

    # ------------------------------------------------------------------
    # Confirm-then-execute (see `_pending_actions` note in __init__)
    # ------------------------------------------------------------------
    async def _execute_pending_action(self, employee_id: str) -> AgentResult:
        pending = self._pending_actions.pop(employee_id)
        tool_name = pending["tool"]
        arguments = dict(pending["arguments"])
        arguments["confirm"] = True  # only ever set True here, from the human-supplied confirm flag

        result = await self._call_tool(tool_name, arguments)
        trace = [ToolCallTrace(tool_name=tool_name, arguments=arguments, result_summary=_summarize_result(tool_name, result))]

        if tool_name == "create_mock_hr_ticket" and result.get("created"):
            answer = f"Done -- I created mock HR ticket {result['ticket']['ticket_id']} as previewed."
        else:
            answer = f"Confirmed and executed: {json.dumps(result)[:300]}"

        return AgentResult(
            answer=answer,
            tool_trace=trace,
            needs_confirmation=False,
            llm_used=False,
            basis=f"Executed the previously previewed, human-confirmed action ({tool_name}) directly; "
                  "no new LLM turn was needed since the action and its arguments were already fixed at "
                  "preview time.",
        )

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------
    @staticmethod
    def _needs_employee_id(message: str, employee_id: str | None) -> str | None:
        if employee_id:
            return None
        lowered = message.lower()
        if any(phrase in lowered for phrase in _NEEDS_EMPLOYEE_ID_PHRASES):
            return (
                "I can help with that, but I need your employee ID first (e.g. \"E1002\") "
                "so I can look up your specific PTO balance, benefits, or profile. "
                "Could you provide it?"
            )
        return None

    @staticmethod
    def _safe_tool_arguments(tool_name: str, arguments: dict[str, Any], confirm_param: bool) -> dict[str, Any]:
        """Never trust the LLM's own `confirm` argument for a write action; only the
        explicit, human-supplied `confirm` field on the API request may set it True."""
        if tool_name == "create_mock_hr_ticket":
            arguments = dict(arguments)
            arguments["confirm"] = bool(confirm_param)
        return arguments

    # ------------------------------------------------------------------
    # LLM-backed agentic loop
    # ------------------------------------------------------------------
    async def _run_llm_loop(self, message: str, employee_id: str | None, confirm: bool) -> AgentResult:
        tool_specs = await self.mcp_client.list_tools()
        anthropic_tools = [t.to_anthropic_tool() for t in tool_specs]

        context_prefix = f"[Context: employee_id={employee_id}]\n" if employee_id else ""
        messages: list[dict[str, Any]] = [{"role": "user", "content": context_prefix + message}]

        trace: list[ToolCallTrace] = []
        citations: list[dict[str, str]] = []
        needs_confirmation = False
        pending_action: dict[str, Any] | None = None

        for _ in range(MAX_LLM_TURNS):
            # Offload the synchronous Anthropic SDK call to a thread: it's a
            # blocking network call, and running it directly on the event loop
            # would stall everything else this process is doing concurrently
            # (including pumping the MCP subprocess's stdio pipes) for the
            # entire request duration.
            response = await asyncio.to_thread(
                self.llm_client.create_message, messages, system=SYSTEM_PROMPT, tools=anthropic_tools
            )

            tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]

            if not tool_use_blocks:
                answer = "\n".join(text_blocks).strip() or "I don't have a response for that."
                return AgentResult(
                    answer=answer,
                    citations=citations,
                    tool_trace=trace,
                    needs_confirmation=needs_confirmation,
                    pending_action=pending_action,
                    llm_used=True,
                    basis="Synthesized by Claude from the tool results listed in tool_trace.",
                )

            messages.append({"role": "assistant", "content": response.content})
            tool_result_blocks = []

            for block in tool_use_blocks:
                safe_args = self._safe_tool_arguments(block.name, block.input, confirm)
                result = await self._call_tool(block.name, safe_args)
                trace.append(
                    ToolCallTrace(
                        tool_name=block.name,
                        arguments=safe_args,
                        result_summary=_summarize_result(block.name, result),
                    )
                )
                citations.extend(_extract_citations(result))

                if block.name == "create_mock_hr_ticket" and isinstance(result, dict) and result.get("needs_confirmation"):
                    needs_confirmation = True
                    pending_action = {"tool": "create_mock_hr_ticket", "arguments": safe_args}
                    if employee_id:
                        self._pending_actions[employee_id] = pending_action

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})

        citations = _dedupe_citations(citations)
        return AgentResult(
            answer="I gathered some information but reached my step limit before finishing. "
                   "Please rephrase your question or ask a more specific follow-up.",
            citations=citations,
            tool_trace=trace,
            needs_confirmation=needs_confirmation,
            pending_action=pending_action,
            llm_used=True,
            basis="Stopped after MAX_LLM_TURNS without a final answer.",
        )

    # ------------------------------------------------------------------
    # Deterministic fallback (no ANTHROPIC_API_KEY)
    # ------------------------------------------------------------------
    async def _run_fallback(
        self,
        message: str,
        employee_id: str | None,
        confirm: bool,
        llm_error: str | None = None,
    ) -> AgentResult:
        lowered = message.lower()
        trace: list[ToolCallTrace] = []
        note = (
            "ANTHROPIC_API_KEY is not set, so this response was produced by a deterministic, "
            "template-based fallback (not real LLM reasoning). RAG retrieval and MCP tools "
            "were still exercised for real. Set ANTHROPIC_API_KEY to enable full LLM synthesis."
        )
        if llm_error:
            note = f"LLM call failed ({llm_error}). " + note

        wants_pto = any(kw in lowered for kw in ["pto", "vacation", "time off", "time-off"])
        wants_remote = "remote" in lowered
        wants_profile = any(kw in lowered for kw in ["my profile", "my role", "my department", "who is my manager", "my hire date"])
        wants_benefits = any(kw in lowered for kw in ["benefit", "enrolled", "medical plan", "401k", "401(k)", "dental", "vision"])

        if wants_pto and employee_id:
            return await self._fallback_pto_workflow(employee_id, message, confirm, note, trace)
        if wants_remote and employee_id:
            return await self._fallback_remote_workflow(employee_id, message, note, trace)
        if wants_profile and employee_id:
            return await self._fallback_profile_lookup(employee_id, note, trace)
        if wants_benefits and employee_id:
            return await self._fallback_benefits_lookup(employee_id, note, trace)

        # Generic RAG-only fallback.
        results = await self._mcp_search(message, trace, k=4)
        if not results["results"] or results["results"][0]["score"] < OUT_OF_SCOPE_SIMILARITY_THRESHOLD:
            return AgentResult(
                answer=(
                    "I couldn't find anything about that in the Northwind HR policy corpus, "
                    "so I don't want to guess. This looks out of scope for this assistant -- "
                    "please contact HR directly or rephrase your question."
                ),
                tool_trace=trace,
                escalated=True,
                llm_used=False,
                basis="No sufficiently relevant policy evidence was retrieved (out-of-corpus guardrail).",
            )

        top = results["results"][0]
        answer = (
            f"Here's what the {top['title']} says under \"{top['section']}\":\n\n"
            f"{top['text']}\n\n({note})"
        )
        return AgentResult(
            answer=answer,
            citations=results["citations"],
            tool_trace=trace,
            llm_used=False,
            basis=f"Top RAG match: {top['doc_id']} > {top['section']} (score={top['score']}).",
        )

    async def _mcp_search(self, query: str, trace: list[ToolCallTrace], k: int = 4) -> dict[str, Any]:
        result = await self._call_tool("search_policy_documents", {"query": query, "k": k})
        trace.append(
            ToolCallTrace(
                tool_name="search_policy_documents",
                arguments={"query": query, "k": k},
                result_summary=_summarize_result("search_policy_documents", result),
            )
        )
        return result

    async def _fallback_pto_workflow(
        self, employee_id: str, message: str, confirm: bool, note: str, trace: list[ToolCallTrace]
    ) -> AgentResult:
        balance_result = await self._call_tool("check_pto_balance", {"employee_id": employee_id})
        trace.append(ToolCallTrace("check_pto_balance", {"employee_id": employee_id}, _summarize_result("check_pto_balance", balance_result)))

        policy_result = await self._mcp_search("PTO accrual balance manager approval requesting time off", trace, k=3)
        citations = list(policy_result["citations"])

        if not balance_result.get("found"):
            return AgentResult(
                answer=f"I couldn't find a PTO record for employee {employee_id!r}. Please double-check the employee ID.",
                tool_trace=trace,
                escalated=True,
                llm_used=False,
                basis="check_pto_balance returned found=False.",
            )

        balance = balance_result["pto_balance_days"]
        answer_lines = [
            f"Your current PTO balance is {balance} days (accrual rate: "
            f"{balance_result['accrual_rate_days_per_month']} days/month).",
            "Per Northwind's PTO Policy, your direct manager must approve PTO requests, "
            "and requests should be submitted at least 5 business days in advance for planned absences.",
        ]

        pending_action = None
        needs_confirmation = False
        if any(kw in message.lower() for kw in ["ticket", "submit", "file a request", "request it", "create a request"]):
            ticket_result = await self._call_tool(
                "create_mock_hr_ticket",
                {
                    "employee_id": employee_id,
                    "subject": "PTO request",
                    "description": message,
                    "category": "pto",
                    "confirm": confirm,
                },
            )
            trace.append(
                ToolCallTrace(
                    "create_mock_hr_ticket",
                    {"employee_id": employee_id, "subject": "PTO request", "category": "pto", "confirm": confirm},
                    _summarize_result("create_mock_hr_ticket", ticket_result),
                )
            )
            if ticket_result.get("needs_confirmation"):
                needs_confirmation = True
                pending_action = {"tool": "create_mock_hr_ticket", "arguments": ticket_result["preview"]}
                self._pending_actions[employee_id] = pending_action
                answer_lines.append(
                    "I've prepared a mock HR ticket for your PTO request but have NOT created it yet -- "
                    "reply with confirm=true to actually create it."
                )
            elif ticket_result.get("created"):
                answer_lines.append(f"I created mock HR ticket {ticket_result['ticket']['ticket_id']} for your PTO request.")

        answer_lines.append(f"\n({note})")
        return AgentResult(
            answer="\n".join(answer_lines),
            citations=citations,
            tool_trace=trace,
            needs_confirmation=needs_confirmation,
            pending_action=pending_action,
            llm_used=False,
            basis="check_pto_balance + PTO policy retrieval; ticket action gated on explicit confirm.",
        )

    async def _fallback_remote_workflow(
        self, employee_id: str, message: str, note: str, trace: list[ToolCallTrace]
    ) -> AgentResult:
        profile_result = await self._call_tool("lookup_employee_profile", {"employee_id": employee_id})
        trace.append(ToolCallTrace("lookup_employee_profile", {"employee_id": employee_id}, _summarize_result("lookup_employee_profile", profile_result)))

        if not profile_result.get("found"):
            return AgentResult(
                answer=f"I couldn't find an employee profile for {employee_id!r}. Please double-check the employee ID.",
                tool_trace=trace,
                escalated=True,
                llm_used=False,
                basis="lookup_employee_profile returned found=False.",
            )

        compliance_result = await self._call_tool(
            "check_policy_compliance", {"scenario": message, "employee_id": employee_id}
        )
        trace.append(
            ToolCallTrace(
                "check_policy_compliance",
                {"scenario": message, "employee_id": employee_id},
                _summarize_result("check_policy_compliance", compliance_result),
            )
        )
        citations = _dedupe_citations(compliance_result.get("citations", []))

        judgment = compliance_result.get("judgment", "insufficient_evidence")
        role = profile_result.get("role", "your role")
        answer = (
            f"Based on your profile ({role}, {profile_result.get('location')}) and the Remote Work "
            f"and Data Security policies, the compliance check for this scenario came back: "
            f"**{judgment.replace('_', ' ')}**.\n\n{compliance_result.get('explanation', '')}\n\n({note})"
        )
        return AgentResult(
            answer=answer,
            citations=citations,
            tool_trace=trace,
            llm_used=False,
            basis=f"check_policy_compliance judgment={judgment} based on retrieved remote-work policy evidence.",
        )

    async def _fallback_profile_lookup(
        self, employee_id: str, note: str, trace: list[ToolCallTrace]
    ) -> AgentResult:
        profile_result = await self._call_tool("lookup_employee_profile", {"employee_id": employee_id})
        trace.append(ToolCallTrace("lookup_employee_profile", {"employee_id": employee_id}, _summarize_result("lookup_employee_profile", profile_result)))

        if not profile_result.get("found"):
            return AgentResult(
                answer=f"I couldn't find an employee profile for {employee_id!r}. Please double-check the employee ID.",
                tool_trace=trace,
                escalated=True,
                llm_used=False,
                basis="lookup_employee_profile returned found=False.",
            )

        answer = (
            f"Here's your profile on file: {profile_result.get('name')}, {profile_result.get('role')} "
            f"in {profile_result.get('department')}, based in {profile_result.get('location')}. "
            f"Employment type: {profile_result.get('employment_type')}. Hire date: {profile_result.get('hire_date')}.\n\n({note})"
        )
        return AgentResult(
            answer=answer,
            tool_trace=trace,
            llm_used=False,
            basis="lookup_employee_profile returned the employee's on-file profile.",
        )

    async def _fallback_benefits_lookup(
        self, employee_id: str, note: str, trace: list[ToolCallTrace]
    ) -> AgentResult:
        benefits_result = await self._call_tool("lookup_benefits_status", {"employee_id": employee_id})
        trace.append(ToolCallTrace("lookup_benefits_status", {"employee_id": employee_id}, _summarize_result("lookup_benefits_status", benefits_result)))

        if not benefits_result.get("found"):
            return AgentResult(
                answer=f"I couldn't find a benefits record for {employee_id!r}. Please double-check the employee ID.",
                tool_trace=trace,
                escalated=True,
                llm_used=False,
                basis="lookup_benefits_status returned found=False.",
            )

        answer = (
            f"Your current benefits elections: medical plan = {benefits_result.get('medical_plan')}, "
            f"dental = {benefits_result.get('dental')}, vision = {benefits_result.get('vision')}, "
            f"401(k) enrolled = {benefits_result.get('retirement_401k_enrolled')} "
            f"({benefits_result.get('retirement_401k_contribution_pct')}% contribution), "
            f"life insurance = {benefits_result.get('life_insurance')}.\n\n({note})"
        )
        return AgentResult(
            answer=answer,
            tool_trace=trace,
            llm_used=False,
            basis="lookup_benefits_status returned the employee's on-file benefits elections.",
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _summarize_result(tool_name: str, result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)[:200]
    if "error" in result and result["error"]:
        return f"error: {result.get('detail', result.get('error'))}"[:200]
    if tool_name == "search_policy_documents":
        n = len(result.get("results", []))
        top = result["results"][0]["doc_id"] if result.get("results") else None
        return f"{n} chunk(s) retrieved; top match: {top}"
    if tool_name == "check_policy_compliance":
        return f"judgment={result.get('judgment')}, flags={result.get('flags')}"
    if tool_name == "check_pto_balance":
        return f"found={result.get('found')}, balance={result.get('pto_balance_days')}"
    if tool_name == "lookup_employee_profile":
        return f"found={result.get('found')}, role={result.get('role')}"
    if tool_name == "lookup_benefits_status":
        return f"found={result.get('found')}, plan={result.get('medical_plan')}"
    if tool_name == "create_mock_hr_ticket":
        if result.get("created"):
            return f"created ticket {result['ticket']['ticket_id']}"
        return "preview only (needs_confirmation=True), nothing written"
    if tool_name == "draft_hr_email":
        return "draft composed, not sent"
    return json.dumps(result)[:200]


def _extract_citations(result: Any) -> list[dict[str, str]]:
    if isinstance(result, dict) and result.get("citations"):
        return result["citations"]
    return []


def _dedupe_citations(citations: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out = []
    for c in citations:
        key = (c.get("doc_id"), c.get("section"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
