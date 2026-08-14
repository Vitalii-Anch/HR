"""
MCP client wrapper: launches mcp_server/server.py as a local subprocess over
stdio and speaks the real MCP protocol to it (initialize, tools/list,
tools/call). The orchestrator never imports mcp_server functions directly --
every tool invocation is a genuine MCP round trip, which is what
tests/test_mcp_tools.py verifies.

Usage:
    client = MCPToolClient()
    await client.connect()
    tools = await client.list_tools()
    result = await client.call_tool("check_pto_balance", {"employee_id": "E1002"})
    await client.close()

or as an async context manager:
    async with MCPToolClient() as client:
        ...
"""
from __future__ import annotations

import json
import os
import shlex
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import settings


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert this MCP tool spec into the shape Claude's tool-use API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class MCPToolClient:
    def __init__(self, command: str | None = None, args: list[str] | None = None):
        self.command = command or settings.mcp_server_command
        self.args = args if args is not None else shlex.split(settings.mcp_server_args)
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools_cache: list[ToolSpec] | None = None

    async def connect(self) -> None:
        if self._session is not None:
            return
        # The MCP SDK's stdio_client, if not given an explicit `env`, spawns
        # the subprocess with only a small hardcoded safe-list of inherited
        # variables (get_default_environment()) -- NOT the parent process's
        # actual environment. That silently drops every one of this
        # project's own config env vars (CHROMA_PERSIST_DIR, CORPUS_DIR,
        # MOCK_DATA_DIR, EMBEDDING_MODEL_PATH, RETRIEVAL_TOP_K, ...) for the
        # MCP server subprocess, which then falls back to app/config.py's
        # hardcoded defaults instead of whatever this process/Render's
        # dashboard actually configured. Passing the real environment
        # through explicitly makes the subprocess's configuration match the
        # parent process's, as intended.
        params = StdioServerParameters(command=self.command, args=self.args, env=dict(os.environ))
        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._session = None
        self._exit_stack = None
        self._tools_cache = None

    async def __aenter__(self) -> "MCPToolClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def list_tools(self, refresh: bool = False) -> list[ToolSpec]:
        """Discover tools dynamically via a real MCP tools/list call (never hardcoded)."""
        if self._tools_cache is not None and not refresh:
            return self._tools_cache
        if self._session is None:
            raise RuntimeError("MCPToolClient is not connected. Call connect() first.")
        result = await self._session.list_tools()
        self._tools_cache = [
            ToolSpec(name=t.name, description=t.description or "", input_schema=t.inputSchema)
            for t in result.tools
        ]
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool by name and return its parsed result (JSON-decoded if possible)."""
        if self._session is None:
            raise RuntimeError("MCPToolClient is not connected. Call connect() first.")
        result = await self._session.call_tool(name, arguments)
        texts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                texts.append(text)
        raw = "\n".join(texts) if texts else None
        if raw is None:
            return {"error": "Tool returned no content."} if result.isError else {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = {"text": raw}
        if result.isError:
            return {"error": True, "detail": parsed}
        return parsed
