"""
Central runtime configuration for the HR Agentic RAG system.

All configuration is read from environment variables (loaded from a local
`.env` file via python-dotenv if present) so that the same code works:
  - in local development (.env file),
  - in CI (no .env, no ANTHROPIC_API_KEY -- LLM calls are mocked in tests),
  - on a Render deployment (real environment variables set in the dashboard).

Nothing in this module requires network access or an API key to import or
instantiate; `ANTHROPIC_API_KEY` is only read lazily wherever the LLM client
actually needs it, never at import time, so the rest of the app (RAG,
mock-data tools, MCP server) works even when no key is configured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if present. Safe no-op if the file is missing.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path = _PROJECT_ROOT

    # LLM (Anthropic)
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    anthropic_model: str = field(default_factory=lambda: _get("ANTHROPIC_MODEL", "claude-sonnet-5"))

    # Embeddings
    embedding_model_name: str = field(default_factory=lambda: _get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"))
    embedding_model_path: str = field(default_factory=lambda: _get("EMBEDDING_MODEL_PATH", "./models/all-MiniLM-L6-v2"))

    # Chroma
    chroma_persist_dir: str = field(default_factory=lambda: _get("CHROMA_PERSIST_DIR", "./data/chroma"))
    chroma_collection_name: str = field(default_factory=lambda: _get("CHROMA_COLLECTION_NAME", "hr_policy_docs"))

    # Retrieval
    retrieval_top_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_TOP_K", 4))

    # Data
    mock_data_dir: str = field(default_factory=lambda: _get("MOCK_DATA_DIR", "./mock_data"))
    corpus_dir: str = field(default_factory=lambda: _get("CORPUS_DIR", "./corpus"))

    # Web app
    app_host: str = field(default_factory=lambda: _get("APP_HOST", "0.0.0.0"))
    app_port: int = field(default_factory=lambda: _get_int("APP_PORT", 8000))

    # MCP server subprocess launch command
    mcp_server_command: str = field(default_factory=lambda: _get("MCP_SERVER_COMMAND", "python"))
    mcp_server_args: str = field(default_factory=lambda: _get("MCP_SERVER_ARGS", "-m mcp_server.server"))

    def resolve(self, rel_path: str) -> Path:
        """Resolve a possibly-relative path against the project root."""
        p = Path(rel_path)
        return p if p.is_absolute() else (self.project_root / p).resolve()


settings = Settings()
