"""Structured trace/response types shared by the orchestrator and the web API.

This is NOT hidden chain-of-thought -- it is a plain, structured record of
which tools were called, with what arguments, and a short summary of what
each returned, plus the citations that back the final answer. Graders (and
the /chat API) can inspect exactly why the agent answered the way it did.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ToolCallTrace:
    tool_name: str
    arguments: dict[str, Any]
    result_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    answer: str
    citations: list[dict[str, str]] = field(default_factory=list)
    tool_trace: list[ToolCallTrace] = field(default_factory=list)
    needs_confirmation: bool = False
    pending_action: dict[str, Any] | None = None
    clarification_needed: bool = False
    escalated: bool = False
    llm_used: bool = False
    basis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "tool_trace": [t.to_dict() for t in self.tool_trace],
            "needs_confirmation": self.needs_confirmation,
            "pending_action": self.pending_action,
            "clarification_needed": self.clarification_needed,
            "escalated": self.escalated,
            "llm_used": self.llm_used,
            "basis": self.basis,
        }
