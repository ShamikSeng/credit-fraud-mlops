# Prepare the live demo: start the stack, then feed real test-set traffic so the drift
# monitor and Grafana show a genuine "normal -> drifted" story. Takes ~2-3 minutes.
#
# Run from the project root, with Docker Desktop running:
#   powershell -ExecutionPolicy Bypass -File demo\prepare_demo.ps1
#
# Add -Build after changing code, to rebuild the app image first.
param([switch]$Build)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\env\Scripts\python.exe"

Write-Host "`n[1/5] Starting app, Prometheus and Grafana..." -ForegroundColor Cyan
if ($Build) { docker compose up -d --build } else { docker compose up -d }
# Restart the app so the drift buffer starts empty and the 500-input windows line up.
docker compose restart app | Out-Null

Write-Host "[2/5] Waiting for the API..." -ForegroundColor Cyan
do {
    Start-Sleep 2
    try { $ok = (Invoke-WebRequest -UseBasicParsing http://localhost:8005/).StatusCode -eq 200 } catch { $ok = $false }
} until ($ok)
# First request loads the model from MLflow (~3 s); do it now, not in front of the panel.
Invoke-RestMethod -Method Post -ContentType application/json -Body (Get-Content demo\legit_example.json -Raw) `
    http://localhost:8005/prediction_api | Out-Null
docker compose restart app | Out-Null   # drop the warm-up row from the drift buffer
do {
    Start-Sleep 2
    try { $ok = (Invoke-WebRequest -UseBasicParsing http://localhost:8005/).StatusCode -eq 200 } catch { $ok = $false }
} until ($ok)
Invoke-WebRequest -UseBasicParsing http://localhost:8005/ui | Out-Null

Write-Host "[3/5] Sending 500 NORMAL test-set transactions (baseline window)..." -ForegroundColor Cyan
& $py demo\send_traffic.py --n 500
if ($LASTEXITCODE -ne 0) { throw "send_traffic.py failed" }

Write-Host "[4/5] Waiting 45 s so Prometheus records the baseline point for Grafana..." -ForegroundColor Cyan
Start-Sleep 45

Write-Host "[5/5] Sending 500 SHIFTED transactions (simulated change in customer behaviour)..." -ForegroundColor Cyan
& $py demo\send_traffic.py --n 500 --shift
if ($LASTEXITCODE -ne 0) { throw "send_traffic.py failed" }

Write-Host "`nReady. Leave this stack running for the demo." -ForegroundColor Green
Write-Host "  Console : http://localhost:8005/ui"
Write-Host "  Grafana : http://localhost:3000/d/fraud-input-drift  (admin / admin, last 15 minutes)"
Write-Host "  Alerts  : http://localhost:9090/alerts"
