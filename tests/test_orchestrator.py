"""
Orchestrator tests, covering:
  - guardrails (missing employee_id -> clarification; out-of-scope -> escalation)
  - the deterministic fallback path for both required demo workflows
  - the LLM-backed agentic loop, with the Anthropic call fully mocked/stubbed
    (never calls the real Anthropic API; no ANTHROPIC_API_KEY needed)

Each test opens its own `async with mcp_client as client:` block (see
conftest.py / test_mcp_tools.py for why) so the whole request lifecycle runs
inside a single coroutine/task.
"""
from types import SimpleNamespace

import pytest

from app.agent.orchestrator import Orchestrator
from app.agent.llm_client import LLMClient


# ---------------------------------------------------------------------
# Fallback-mode (no LLM key) tests
# ---------------------------------------------------------------------

async def test_missing_employee_id_triggers_clarification(mcp_client):
    async with mcp_client as client:
        orch = Orchestrator(client)
        result = await orch.handle_message("What is my PTO balance?")
    assert result.clarification_needed is True
    assert "employee id" in result.answer.lower()


async def test_out_of_scope_triggers_escalation(mcp_client):
    async with mcp_client as client:
        orch = Orchestrator(client)
        result = await orch.handle_message("What's the weather like today?")
    assert result.escalated is True
    assert result.llm_used is False


async def test_pto_workflow_fallback(mcp_client):
    async with mcp_client as client:
        orch = Orchestrator(client)
        result = await orch.handle_message(
            "What is my PTO balance and can you submit a ticket for a PTO request?",
            employee_id="E1002",
        )
    tool_names = [t.tool_name for t in result.tool_trace]
    assert "check_pto_balance" in tool_names
    assert "search_policy_documents" in tool_names
    assert "create_mock_hr_ticket" in tool_names
    assert result.needs_confirmation is True  # confirm was not set -> preview only
    assert len(result.citations) > 0


async def test_pto_ticket_creation_requires_explicit_confirm(mcp_client):
    async with mcp_client as client:
        orch = Orchestrator(client)
        result = await orch.handle_message(
            "Please submit a ticket for my PTO request.",
            employee_id="E1002",
            confirm=True,
        )
    ticket_calls = [t for t in result.tool_trace if t.tool_name == "create_mock_hr_ticket"]
    assert len(ticket_calls) == 1
    assert ticket_calls[0].arguments["confirm"] is True
    assert "created ticket" in ticket_calls[0].result_summary


async def test_remote_work_workflow_fallback(mcp_client):
    async with mcp_client as client:
        orch = Orchestrator(client)
        result = await orch.handle_message(
            "Can I work remotely from Mexico for a few months?",
            employee_id="E1002",
        )
    tool_names = [t.tool_name for t in result.tool_trace]
    assert "lookup_employee_profile" in tool_names
    assert "check_policy_compliance" in tool_names
    assert len(result.citations) > 0


# ---------------------------------------------------------------------
# LLM-mode tests with a fully stubbed/mocked LLM client (no real API call)
# ---------------------------------------------------------------------

def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


class StubLLMClient(LLMClient):
    """A fake LLMClient that returns a pre-scripted sequence of responses,
    never calling the real Anthropic SDK. Used to exercise the orchestrator's
    tool-use loop deterministically and without any API key."""

    def __init__(self, scripted_responses):
        super().__init__(model="stub-model")
        self._responses = list(scripted_responses)
        self.calls = []

    def is_configured(self) -> bool:
        return True

    def create_message(self, messages, system=None, tools=None, max_tokens=1024):
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        return self._responses.pop(0)


async def test_llm_loop_executes_tool_then_returns_final_answer(mcp_client):
    scripted = [
        SimpleNamespace(content=[_tool_use_block("check_pto_balance", {"employee_id": "E1002"})], stop_reason="tool_use"),
        SimpleNamespace(content=[_text_block("You have 14 days of PTO. (PTO Policy > Eligibility and Accrual)")], stop_reason="end_turn"),
    ]
    stub = StubLLMClient(scripted)

    async with mcp_client as client:
        orch = Orchestrator(client, llm_client=stub)
        result = await orch.handle_message("What is my PTO balance?", employee_id="E1002")

    assert result.llm_used is True
    assert "14" in result.answer
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0].tool_name == "check_pto_balance"
    assert len(stub.calls) == 2  # one turn producing tool_use, one producing the final answer


async def test_llm_loop_never_trusts_llm_confirm_flag(mcp_client):
    """Even if the LLM tries to set confirm=True on create_mock_hr_ticket, the
    orchestrator must override it with the human-supplied `confirm` param."""
    scripted = [
        SimpleNamespace(
            content=[_tool_use_block("create_mock_hr_ticket", {
                "employee_id": "E1002", "subject": "PTO", "description": "x", "confirm": True,
            })],
            stop_reason="tool_use",
        ),
        SimpleNamespace(content=[_text_block("Done.")], stop_reason="end_turn"),
    ]
    stub = StubLLMClient(scripted)

    async with mcp_client as client:
        orch = Orchestrator(client, llm_client=stub)
        # Human did NOT pass confirm=True on the API call.
        result = await orch.handle_message("Create a PTO ticket for me", employee_id="E1002", confirm=False)

    ticket_call = result.tool_trace[0]
    assert ticket_call.arguments["confirm"] is False
    assert "preview only" in ticket_call.result_summary
