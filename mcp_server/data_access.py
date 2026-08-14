"""
Read/write access to the mock structured HR data (mock_data/*.json).

This is intentionally simple file-based access (no database) since the data
is small, synthetic, and read-mostly. `create_mock_hr_ticket` is the only
writer, and it always appends to mock_data/tickets.json -- every write is
labeled as a MOCK ACTION in logs, and no other tool ever mutates data. There
are no real side effects anywhere in this module (no real email is sent, no
real ticketing system is contacted).
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings

logger = logging.getLogger("hr_agentic_rag.mock_data")

# Small in-process cache for the read-only mock datasets (employees, PTO
# balances, benefits -- never written to at runtime). Re-reading and
# JSON-parsing these small files from disk on every single tool call is
# unnecessary I/O; on a throttled free-tier host every bit of avoidable
# per-request I/O adds latency risk, and this mirrors the caching already
# applied to the embedding model and the Chroma client (see
# app/rag/embeddings.py, app/rag/ingest.py). `tickets.json` is deliberately
# NOT cached here since create_ticket() writes to it at runtime and callers
# must always see the current on-disk state.
_READ_ONLY_CACHE: dict[str, list[dict]] = {}
_READ_ONLY_CACHE_LOCK = threading.Lock()
_READ_ONLY_FILES = {"employees.json", "pto_balances.json", "benefits.json"}


def _data_dir() -> Path:
    return settings.resolve(settings.mock_data_dir)


def _read_json_from_disk(filename: str) -> list[dict]:
    path = _data_dir() / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json(filename: str) -> list[dict]:
    if filename not in _READ_ONLY_FILES:
        return _read_json_from_disk(filename)
    if filename not in _READ_ONLY_CACHE:
        with _READ_ONLY_CACHE_LOCK:
            if filename not in _READ_ONLY_CACHE:
                _READ_ONLY_CACHE[filename] = _read_json_from_disk(filename)
    return _READ_ONLY_CACHE[filename]


def _save_json(filename: str, data: list[dict]) -> None:
    path = _data_dir() / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_employees() -> list[dict]:
    return _load_json("employees.json")


def get_employee(employee_id: str) -> dict | None:
    for emp in list_employees():
        if emp["employee_id"].lower() == employee_id.lower():
            return emp
    return None


def get_pto_balance(employee_id: str) -> dict | None:
    for row in _load_json("pto_balances.json"):
        if row["employee_id"].lower() == employee_id.lower():
            return row
    return None


def get_benefits(employee_id: str) -> dict | None:
    for row in _load_json("benefits.json"):
        if row["employee_id"].lower() == employee_id.lower():
            return row
    return None


def list_tickets() -> list[dict]:
    return _load_json("tickets.json")


def create_ticket(
    employee_id: str,
    subject: str,
    description: str,
    category: str = "general",
) -> dict:
    """Append a new MOCK HR ticket to mock_data/tickets.json and return it.

    This is a mock action with no real side effects: it does not contact any
    real ticketing system, and never touches real employee data (the mock
    corpus is entirely synthetic). Every call is logged clearly as a mock
    action for auditability.
    """
    tickets = _load_json("tickets.json")
    ticket = {
        "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
        "employee_id": employee_id,
        "subject": subject,
        "description": description,
        "category": category,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tickets.append(ticket)
    _save_json("tickets.json", tickets)
    logger.info("MOCK ACTION: created mock HR ticket %s for employee %s (no real system contacted)", ticket["ticket_id"], employee_id)
    return ticket
