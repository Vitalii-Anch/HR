#!/usr/bin/env bash
# Demo workflow (a): PTO request guidance.
# Reproduces, via plain curl, the multi-step PTO workflow described in
# design-and-evaluation.md: check PTO balance -> retrieve PTO policy ->
# identify manager approval requirement -> (with confirmation) draft a mock
# HR ticket.
#
# Usage:
#   1. In one terminal: uvicorn app.main:app --host 0.0.0.0 --port 8000
#   2. In another terminal: bash scripts/demo_pto.sh
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "== Step 0: health check =="
curl -sS "$BASE_URL/health"; echo

echo
echo "== Step 1: ask for PTO guidance + request a ticket, WITHOUT confirmation =="
curl -sS -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my PTO balance, and can you submit a ticket to request PTO from Sept 8 to Sept 10 (3 days)?", "employee_id": "E1002"}' \
  | python3 -m json.tool 2>/dev/null || cat

echo
echo "== Step 2: confirm the pending ticket creation (confirm=true) =="
curl -sS -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Yes, please submit that PTO ticket.", "employee_id": "E1002", "confirm": true}' \
  | python3 -m json.tool 2>/dev/null || cat
