# Demo workflow (a): PTO request guidance.
# PowerShell equivalent of demo_pto.sh -- reproduces the multi-step PTO
# workflow described in design-and-evaluation.md: check PTO balance ->
# retrieve PTO policy -> identify manager approval requirement -> (with
# confirmation) draft a mock HR ticket.
#
# Usage:
#   1. In one terminal: uvicorn app.main:app --host 0.0.0.0 --port 8000
#   2. In another terminal: .\scripts\demo_pto.ps1

$BaseUrl = if ($env:BASE_URL) { $env:BASE_URL } else { "http://localhost:8000" }

Write-Host "== Step 0: health check =="
Invoke-RestMethod -Uri "$BaseUrl/health" | ConvertTo-Json

Write-Host "`n== Step 1: ask for PTO guidance + request a ticket, WITHOUT confirmation =="
$body1 = @{
    message     = "What is my PTO balance, and can you submit a ticket to request PTO from Sept 8 to Sept 10 (3 days)?"
    employee_id = "E1002"
} | ConvertTo-Json
$resp1 = Invoke-RestMethod -Uri "$BaseUrl/chat" -Method Post -ContentType "application/json" -Body $body1
$resp1 | ConvertTo-Json -Depth 10

Write-Host "`n== Step 2: confirm the pending ticket creation (confirm=true) =="
$body2 = @{
    message     = "Yes, please submit that PTO ticket."
    employee_id = "E1002"
    confirm     = $true
} | ConvertTo-Json
$resp2 = Invoke-RestMethod -Uri "$BaseUrl/chat" -Method Post -ContentType "application/json" -Body $body2
$resp2 | ConvertTo-Json -Depth 10
