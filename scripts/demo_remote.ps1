# Demo workflow (b): Remote work eligibility.
# PowerShell equivalent of demo_remote.sh -- reproduces the remote-work-
# eligibility workflow described in design-and-evaluation.md: look up
# employee profile -> retrieve remote work / security / tax-location
# policies -> check compliance -> cited next steps.
#
# Usage:
#   1. In one terminal: uvicorn app.main:app --host 0.0.0.0 --port 8000
#   2. In another terminal: .\scripts\demo_remote.ps1

$BaseUrl = if ($env:BASE_URL) { $env:BASE_URL } else { "http://localhost:8000" }

Write-Host "== Step 0: health check =="
Invoke-RestMethod -Uri "$BaseUrl/health" | ConvertTo-Json

Write-Host "`n== Step 1: ask about remote work eligibility for a specific employee =="
$body1 = @{
    message     = "Can I work remotely from Mexico for a few months?"
    employee_id = "E1002"
} | ConvertTo-Json
$resp1 = Invoke-RestMethod -Uri "$BaseUrl/chat" -Method Post -ContentType "application/json" -Body $body1
$resp1 | ConvertTo-Json -Depth 10

Write-Host "`n== Step 2: ambiguous request (no employee id) triggers a clarifying question =="
$body2 = @{
    message = "Am I eligible to work remotely?"
} | ConvertTo-Json
$resp2 = Invoke-RestMethod -Uri "$BaseUrl/chat" -Method Post -ContentType "application/json" -Body $body2
$resp2 | ConvertTo-Json -Depth 10

Write-Host "`n== Step 3: out-of-scope request triggers a refusal, not a hallucinated answer =="
$body3 = @{
    message = "What's the best pizza topping?"
} | ConvertTo-Json
$resp3 = Invoke-RestMethod -Uri "$BaseUrl/chat" -Method Post -ContentType "application/json" -Body $body3
$resp3 | ConvertTo-Json -Depth 10
