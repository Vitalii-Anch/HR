#!/usr/bin/env bash
# Demo workflow (b): Remote work eligibility.
# Reproduces, via plain curl, the remote-work-eligibility workflow described
# in design-and-evaluation.md: look up employee profile -> retrieve remote
# work / security / tax-location policies -> check compliance -> cited next
# steps.
#
# Usage:
#   1. In one terminal: uvicorn app.main:app --host 0.0.0.0 --port 8000
#   2. In another terminal: bash scripts/demo_remote.sh
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "== Step 0: health check =="
curl -sS "$BASE_URL/health"; echo

echo
echo "== Step 1: ask about remote work eligibility for a specific employee =="
curl -sS -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Can I work remotely from Mexico for a few months?", "employee_id": "E1002"}' \
  | python3 -m json.tool 2>/dev/null || cat

echo
echo "== Step 2: ambiguous request (no employee id) triggers a clarifying question =="
curl -sS -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Am I eligible to work remotely?"}' \
  | python3 -m json.tool 2>/dev/null || cat

echo
echo "== Step 3: out-of-scope request triggers a refusal, not a hallucinated answer =="
curl -sS -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What'"'"'s the best pizza topping?"}' \
  | python3 -m json.tool 2>/dev/null || cat
