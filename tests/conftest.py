"""Shared pytest fixtures.

Nothing here requires ANTHROPIC_API_KEY. The `ensure_index` fixture makes
sure the Chroma index exists before any RAG/MCP tests run (idempotent
rebuild from corpus/), and `mcp_client` provides a connected MCPToolClient
backed by a real subprocess of mcp_server/server.py over stdio.
"""
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Make sure a stray ANTHROPIC_API_KEY in the test environment never sneaks
# into these tests -- they must all pass without a real key.
os.environ.pop("ANTHROPIC_API_KEY", None)

from app.rag.ingest import build_index, get_collection  # noqa: E402
from app.agent.mcp_client import MCPToolClient  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_index():
    collection = get_collection(create=True)
    if collection.count() == 0:
        build_index()
    yield


def pytest_sessionfinish(session, exitstatus):
    """Force-exit after the test session completes.

    Each MCP-backed test spawns a subprocess (mcp_server/server.py) over
    stdio. Occasionally a background thread owned by one of the heavier
    native dependencies (torch / sentence-transformers / grpc, pulled in via
    chromadb) keeps the interpreter alive after all tests have already
    passed and normal teardown has finished, which otherwise makes the
    process (and CI) hang well past test completion. Since pytest has
    already reported results and written any output by the time this hook
    runs, it's safe to hard-exit immediately with the real exit status.
    """
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if exitstatus == 0 else exitstatus)


@pytest_asyncio.fixture(scope="session")
async def _connected_mcp_client():
    """One MCP client (one subprocess, one stdio session) shared for the
    whole test session -- spawning a fresh subprocess per test is correct
    but slow (each import of torch/sentence-transformers/chromadb inside the
    subprocess costs a couple of seconds). Session-scoped, paired with a
    session-scoped asyncio loop for both fixtures and tests (see pytest.ini:
    asyncio_default_fixture_loop_scope / asyncio_default_test_loop_scope),
    so this client's connection and every test's usage of it run on the same
    event loop and the same Task, satisfying anyio's requirement that a
    cancel scope be entered/exited from the same Task.

    Deliberately has no teardown: pytest_sessionfinish() below force-exits
    the interpreter after the session completes, which also reaps the
    subprocess. Explicitly closing it here previously triggered an
    anyio "different task" error in some pytest-asyncio/anyio combinations.
    """
    client = MCPToolClient()
    await client.connect()
    return client


@pytest_asyncio.fixture
async def mcp_client(_connected_mcp_client):
    """Yields the shared, already-connected client wrapped so existing
    `async with mcp_client as client:` usage in tests keeps working (the
    context manager here is a no-op passthrough; connect/close lifecycle is
    owned by the session-scoped fixture above)."""

    class _PassthroughCtx:
        async def __aenter__(self_inner):
            return _connected_mcp_client

        async def __aexit__(self_inner, *exc):
            return False

    return _PassthroughCtx()
