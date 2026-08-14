"""
MCP server exposing HR RAG + mock-data tools over stdio transport.

Run standalone for manual testing:
    python -m mcp_server.server

Normally this module is launched as a subprocess by the agent orchestrator's
MCP client (see app/agent/mcp_client.py), which speaks the MCP protocol over
stdio -- the orchestrator never calls these Python functions directly; it
always goes through a real MCP `tools/list` + `tools/call` round trip, which
is what the CI test in tests/test_mcp_tools.py verifies.

Why stdio transport: this project deploys as a single Render free-tier web
service. Running the MCP server as a local subprocess of the same process
that hosts the FastAPI app (rather than as a separate networked service)
avoids needing a second deployed service, a second URL, or any inter-service
auth -- it is the simplest architecture that still genuinely goes through the
MCP protocol rather than calling tool functions directly in-process.

Tool inventory (also documented with full JSON schemas in
mcp_server/README.md):
  1. search_policy_documents   - RAG semantic search over the policy corpus
  2. get_policy_section        - fetch a specific document/section by id
  3. lookup_employee_profile   - mock employee directory lookup
  4. check_pto_balance         - mock PTO balance lookup
  5. lookup_benefits_status    - mock benefits elections lookup
  6. create_mock_hr_ticket     - mock ticket creation (requires confirm=True)
  7. draft_hr_email            - mock email draft (never sends anything)
  8. check_policy_compliance   - RAG-grounded compliance judgment + citations
"""
from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server import data_access
from app.rag.retrieval import retrieve, format_citations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hr_agentic_rag.mcp_server")

mcp = FastMCP("hr-policy-mcp-server")


# ---------------------------------------------------------------------------
# RAG-backed tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_policy_documents(query: str, k: int = 4) -> dict[str, Any]:
    """Semantically search the HR policy corpus (RAG) and return the top-k
    matching chunks with citation metadata (doc_id, title, section).

    Args:
        query: Natural-language question or topic to search the policy corpus for.
        k: Number of chunks to return (default 4).
    """
    results = retrieve(query, k=k)
    return {
        "query": query,
        "results": [
            {
                "doc_id": r.doc_id,
                "title": r.title,
                "section": r.section,
                "text": r.text,
                "score": round(r.score, 4),
                "source_format": r.source_format,
            }
            for r in results
        ],
        "citations": format_citations(results),
    }


@mcp.tool()
def get_policy_section(doc_id: str, section: str | None = None) -> dict[str, Any]:
    """Fetch chunk(s) belonging to a specific policy document (and optionally
    a specific section heading within it), by exact doc_id.

    Args:
        doc_id: The policy document's citation id, e.g. "pto-policy".
        section: Optional section heading (case-insensitive substring match).
    """
    from app.rag.ingest import get_collection

    collection = get_collection(create=True)
    if collection.count() == 0:
        return {"doc_id": doc_id, "section": section, "chunks": [], "error": "Index is empty."}

    where: dict[str, Any] = {"doc_id": doc_id}
    got = collection.get(where=where, include=["documents", "metadatas"])
    docs = got.get("documents", [])
    metas = got.get("metadatas", [])

    chunks = []
    for text, meta in zip(docs, metas):
        if section and section.lower() not in (meta.get("section") or "").lower():
            continue
        chunks.append({"section": meta.get("section"), "title": meta.get("title"), "text": text})

    if not chunks and not section:
        return {"doc_id": doc_id, "section": section, "chunks": [], "error": f"No document found with doc_id={doc_id!r}."}

    return {"doc_id": doc_id, "section": section, "chunks": chunks}


@mcp.tool()
def check_policy_compliance(scenario: str, employee_id: str | None = None) -> dict[str, Any]:
    """Check a described scenario against retrieved HR policy text (RAG) and
    return a compliance judgment with supporting citations.

    This tool always grounds its judgment in retrieved policy chunks (never
    invents policy). Without an LLM available, the judgment is a transparent,
    rule-based heuristic over the retrieved text (flags approval/eligibility
    keywords) rather than free-form reasoning; the orchestrator's LLM layer
    (when ANTHROPIC_API_KEY is configured) uses these same citations to
    produce a fuller natural-language judgment.

    Args:
        scenario: A plain-language description of the situation to check, e.g.
            "Employee wants to work remotely from Mexico for 3 months."
        employee_id: Optional employee id, for cross-referencing role/location.
    """
    results = retrieve(scenario, k=5)
    # Threshold calibrated for TF-IDF cosine-similarity scores, which run
    # lower than the dense-embedding scores this was originally tuned for
    # (see app/agent/orchestrator.py's OUT_OF_SCOPE_SIMILARITY_THRESHOLD for
    # the full explanation -- same recalibration applies here). retrieve()
    # already returns [] outright for zero lexical overlap.
    if not results or results[0].score < 0.12:
        return {
            "scenario": scenario,
            "judgment": "insufficient_evidence",
            "explanation": "No sufficiently relevant policy text was found for this scenario. "
                           "This may be an out-of-scope request; escalate to an HR business partner.",
            "citations": [],
        }

    combined_text = " ".join(r.text.lower() for r in results)
    flags = []
    if "approval" in combined_text or "approve" in combined_text:
        flags.append("requires_approval")
    if "not eligible" in combined_text or "not eligible" in scenario.lower():
        flags.append("possible_ineligibility")
    if "exception" in combined_text:
        flags.append("exception_process_may_apply")
    if "prohibited" in combined_text or "violation" in combined_text:
        flags.append("possible_policy_violation")

    if "possible_policy_violation" in flags or "possible_ineligibility" in flags:
        judgment = "likely_not_compliant_or_ineligible"
    elif "requires_approval" in flags:
        judgment = "compliant_with_conditions"
    else:
        judgment = "likely_compliant"

    employee = data_access.get_employee(employee_id) if employee_id else None

    return {
        "scenario": scenario,
        "employee": employee,
        "judgment": judgment,
        "flags": flags,
        "explanation": (
            "Heuristic judgment derived from retrieved policy text below. "
            "For a full natural-language explanation, this tool's output is "
            "synthesized by the orchestrator's LLM layer when available."
        ),
        "evidence": [{"doc_id": r.doc_id, "section": r.section, "text": r.text} for r in results],
        "citations": format_citations(results),
    }


# ---------------------------------------------------------------------------
# Mock-structured-data tools
# ---------------------------------------------------------------------------

@mcp.tool()
def lookup_employee_profile(employee_id: str) -> dict[str, Any]:
    """Look up a mock employee's profile (name, role, department, manager,
    location, employment type, hire date) by employee_id.

    Args:
        employee_id: The employee's id, e.g. "E1002".
    """
    employee = data_access.get_employee(employee_id)
    if not employee:
        return {"employee_id": employee_id, "found": False, "error": f"No employee found with id {employee_id!r}."}
    return {"found": True, **employee}


@mcp.tool()
def check_pto_balance(employee_id: str) -> dict[str, Any]:
    """Look up a mock employee's current PTO balance and accrual rate.

    Args:
        employee_id: The employee's id, e.g. "E1002".
    """
    balance = data_access.get_pto_balance(employee_id)
    if not balance:
        return {"employee_id": employee_id, "found": False, "error": f"No PTO record found for employee id {employee_id!r}."}
    return {"found": True, **balance}


@mcp.tool()
def lookup_benefits_status(employee_id: str) -> dict[str, Any]:
    """Look up a mock employee's benefits elections and eligibility.

    Args:
        employee_id: The employee's id, e.g. "E1002".
    """
    benefits = data_access.get_benefits(employee_id)
    if not benefits:
        return {"employee_id": employee_id, "found": False, "error": f"No benefits record found for employee id {employee_id!r}."}
    return {"found": True, **benefits}


@mcp.tool()
def create_mock_hr_ticket(
    employee_id: str,
    subject: str,
    description: str,
    category: str = "general",
    confirm: bool = False,
) -> dict[str, Any]:
    """Create a MOCK HR ticket (no real ticketing system is contacted; the
    ticket is appended to a local JSON file for this course project). This
    is a write action and REQUIRES explicit confirmation.

    Args:
        employee_id: The employee id the ticket is being filed for.
        subject: Short ticket subject line.
        description: Full ticket description/body.
        category: Ticket category, e.g. "pto", "benefits", "general".
        confirm: Must be True to actually create the ticket. If False, this
            tool returns a preview of what WOULD be created without writing
            anything, so the caller (agent/orchestrator) can show the user
            the pending action and obtain explicit confirmation first.
    """
    if not confirm:
        return {
            "created": False,
            "needs_confirmation": True,
            "preview": {
                "employee_id": employee_id,
                "subject": subject,
                "description": description,
                "category": category,
            },
            "message": "Ticket not created. Re-call this tool with confirm=true to actually create it.",
        }

    ticket = data_access.create_ticket(employee_id, subject, description, category)
    return {"created": True, "needs_confirmation": False, "ticket": ticket}


@mcp.tool()
def draft_hr_email(
    to_employee_id: str,
    subject: str,
    key_points: str,
    from_name: str = "HR Assistant",
) -> dict[str, Any]:
    """Draft (but never send) an HR-related email. Returns draft text only;
    this tool has no real side effects and does not contact any mail system.

    Args:
        to_employee_id: Employee id the email would be addressed to.
        subject: Email subject line.
        key_points: Plain-language summary of what the email should cover;
            used to compose the draft body.
        from_name: Signature name for the draft (default "HR Assistant").
    """
    employee = data_access.get_employee(to_employee_id)
    greeting_name = employee["name"].split(" ")[0] if employee else to_employee_id

    body = (
        f"Hi {greeting_name},\n\n"
        f"{key_points.strip()}\n\n"
        f"Please let me know if you have any questions.\n\n"
        f"Best,\n{from_name}"
    )
    draft = {"to_employee_id": to_employee_id, "subject": subject, "body": body}
    logger.info("MOCK ACTION: drafted (not sent) HR email to %s, subject=%r", to_employee_id, subject)
    return {"draft": draft, "sent": False, "note": "This is a draft only. No email was sent."}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
