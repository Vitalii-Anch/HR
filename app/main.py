"""
FastAPI web app: minimal chat UI + /chat + /health endpoints.

This module wires together the RAG index, the MCP client (which launches
mcp_server/server.py as a subprocess over stdio), and the agent orchestrator.
See design-and-evaluation.md for the full architecture diagram.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.config import settings
from app.agent.mcp_client import MCPToolClient
from app.agent.orchestrator import Orchestrator
from app.rag.ingest import build_index, get_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hr_agentic_rag.main")

_TEMPLATE_PATH = Path(__file__).parent / "web" / "templates" / "chat.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure the RAG index exists (idempotent; builds from corpus/ if empty).
    try:
        collection = get_collection(create=True)
        if collection.count() == 0:
            logger.info("Chroma collection is empty; building index from corpus/ ...")
            stats = build_index()
            logger.info("Indexed %s documents into %s chunks.", stats.documents, stats.chunks)
        app.state.index_loaded = collection.count() > 0
    except Exception:
        logger.exception("Failed to initialize the RAG index.")
        app.state.index_loaded = False

    # Launch the MCP server subprocess and connect over stdio.
    mcp_client = MCPToolClient()
    try:
        await mcp_client.connect()
        app.state.mcp_connected = True
    except Exception:
        logger.exception("Failed to connect to the MCP server.")
        app.state.mcp_connected = False

    app.state.mcp_client = mcp_client
    app.state.orchestrator = Orchestrator(mcp_client)

    yield

    try:
        await mcp_client.close()
    except Exception:
        logger.exception("Error while closing the MCP client.")


app = FastAPI(title="Northwind HR Agentic RAG", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    employee_id: str | None = None
    confirm: bool | None = False


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    tool_trace: list[dict]
    needs_confirmation: bool = False
    pending_action: dict | None = None
    clarification_needed: bool = False
    escalated: bool = False
    llm_used: bool = False
    basis: str = ""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mcp_connected": bool(getattr(app.state, "mcp_connected", False)),
        "index_loaded": bool(getattr(app.state, "index_loaded", False)),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    orchestrator: Orchestrator = app.state.orchestrator
    result = await orchestrator.handle_message(
        message=req.message,
        employee_id=req.employee_id,
        confirm=bool(req.confirm),
    )
    return ChatResponse(**result.to_dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False)
