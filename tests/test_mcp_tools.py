"""
(b) MCP tool discovery test + (c) MCP tool call test.

These go through the REAL MCP protocol (stdio subprocess running
mcp_server/server.py), not direct Python function calls, per project
requirements. No ANTHROPIC_API_KEY is needed for any of this.

Each test connects its own MCPToolClient via `async with` so that the
connect/use/close lifecycle runs inside a single coroutine (see conftest.py
for why -- this avoids an anyio cross-task cancel-scope error that occurs
with some pytest-asyncio/anyio version combinations when a client is opened
in fixture setup and closed in fixture teardown).
"""
import pytest

EXPECTED_TOOL_NAMES = {
    "search_policy_documents",
    "get_policy_section",
    "lookup_employee_profile",
    "check_pto_balance",
    "lookup_benefits_status",
    "create_mock_hr_ticket",
    "draft_hr_email",
    "check_policy_compliance",
}


async def test_mcp_tool_discovery(mcp_client):
    async with mcp_client as client:
        tools = await client.list_tools()
    names = {t.name for t in tools}
    assert EXPECTED_TOOL_NAMES.issubset(names), f"Missing tools: {EXPECTED_TOOL_NAMES - names}"
    for t in tools:
        assert t.description, f"Tool {t.name} has no description"
        assert "properties" in t.input_schema


async def test_check_pto_balance_known_employee(mcp_client):
    async with mcp_client as client:
        result = await client.call_tool("check_pto_balance", {"employee_id": "E1002"})
    assert result["found"] is True
    assert isinstance(result["pto_balance_days"], (int, float))
    assert result["pto_balance_days"] >= 0


async def test_check_pto_balance_unknown_employee(mcp_client):
    async with mcp_client as client:
        result = await client.call_tool("check_pto_balance", {"employee_id": "E9999"})
    assert result["found"] is False


async def test_lookup_employee_profile(mcp_client):
    async with mcp_client as client:
        result = await client.call_tool("lookup_employee_profile", {"employee_id": "E1002"})
    assert result["found"] is True
    assert result["role"] == "Software Engineer"


async def test_lookup_benefits_status(mcp_client):
    async with mcp_client as client:
        result = await client.call_tool("lookup_benefits_status", {"employee_id": "E1002"})
    assert result["found"] is True
    assert "medical_plan" in result


async def test_search_policy_documents(mcp_client):
    async with mcp_client as client:
        result = await client.call_tool("search_policy_documents", {"query": "PTO accrual rate", "k": 3})
    assert len(result["results"]) > 0
    assert result["results"][0]["doc_id"] == "pto-policy"
    assert len(result["citations"]) > 0


async def test_check_policy_compliance(mcp_client):
    async with mcp_client as client:
        result = await client.call_tool(
            "check_policy_compliance",
            {"scenario": "Employee wants to work remotely from another country for 3 months.", "employee_id": "E1002"},
        )
    assert "judgment" in result
    assert len(result["citations"]) > 0


async def test_create_mock_hr_ticket_requires_confirmation(mcp_client):
    async with mcp_client as client:
        preview = await client.call_tool(
            "create_mock_hr_ticket",
            {"employee_id": "E1002", "subject": "Test", "description": "Test ticket", "confirm": False},
        )
        assert preview["created"] is False
        assert preview["needs_confirmation"] is True

        created = await client.call_tool(
            "create_mock_hr_ticket",
            {"employee_id": "E1002", "subject": "Test", "description": "Test ticket", "confirm": True},
        )
    assert created["created"] is True
    assert created["ticket"]["ticket_id"].startswith("TCK-")


async def test_draft_hr_email_never_sends(mcp_client):
    async with mcp_client as client:
        result = await client.call_tool(
            "draft_hr_email",
            {"to_employee_id": "E1002", "subject": "Benefits reminder", "key_points": "Open enrollment ends soon."},
        )
    assert result["sent"] is False
    assert "draft" in result
