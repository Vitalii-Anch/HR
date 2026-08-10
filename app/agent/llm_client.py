"""
Thin wrapper around the Anthropic Claude SDK.

All LLM calls in this project go through this one class so that:
  - ANTHROPIC_API_KEY is read from the environment lazily (at call time),
    never at import time -- importing this module, or even instantiating
    LLMClient, never requires a key.
  - Tests can substitute a fake/mock object satisfying the same interface
    (`is_configured()`, `create_message(...)`) instead of monkeypatching the
    Anthropic SDK internals, keeping tests fast and key-free.
"""
from __future__ import annotations

import os
from typing import Any


class LLMNotConfiguredError(RuntimeError):
    """Raised when an LLM call is attempted but ANTHROPIC_API_KEY is not set."""


class LLMClient:
    def __init__(self, model: str | None = None):
        from app.config import settings

        self.model = model or settings.anthropic_model
        self._client = None  # lazily constructed, only once a key is confirmed present

    def is_configured(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def _get_client(self):
        if self._client is None:
            if not self.is_configured():
                raise LLMNotConfiguredError(
                    "ANTHROPIC_API_KEY is not set. LLM-backed synthesis is unavailable; "
                    "RAG retrieval and MCP tools still work without it."
                )
            import anthropic

            # Explicit timeout: the SDK's default is 10 minutes, which would make a
            # genuinely hung/stalled request indistinguishable from the app itself
            # being broken (and would block the whole asyncio event loop for that
            # entire window, since this call is synchronous -- see create_message
            # below). 45s is generous for a single tool-use turn against a small
            # tool schema and short policy chunks.
            self._client = anthropic.Anthropic(timeout=45.0, max_retries=2)  # reads ANTHROPIC_API_KEY from env
        return self._client

    def create_message(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> Any:
        """Call the Claude Messages API. Raises LLMNotConfiguredError if no key is set."""
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return client.messages.create(**kwargs)
