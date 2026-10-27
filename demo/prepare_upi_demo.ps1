# Prepare the UPI part of the demo: start the stack, then send real simulated UPI payments
# (normal months, then the festive season) so the drift monitor and Grafana show a genuine
# "normal -> drifted" story. Takes ~2 minutes.
#
# Run from the project root, with Docker Desktop running:
#   powershell -ExecutionPolicy Bypass -File demo\prepare_upi_demo.ps1
#
# Add -Build after changing code, to rebuild the image first.
param([switch]$Build)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\env\Scripts\python.exe"

function Wait-Api($url) {
    do {
        Start-Sleep 2
        try { $ok = (Invoke-WebRequest -UseBasicParsing $url).StatusCode -eq 200 } catch { $ok = $false }
    } until ($ok)
}

if (-not (Test-Path "prediction_model\monitoring\upi_reference_data.csv")) {
    Write-Host "`n[setup] First run: simulating the UPI stream and training the UPI model..." -ForegroundColor Cyan
    & $py -m prediction_model.upi.simulate
    & $py -m prediction_model.upi.train
}

Write-Host "`n[1/4] Starting the stack (ULB app, UPI app, Prometheus, Grafana)..." -ForegroundColor Cyan
if ($Build) { docker compose up -d --build } else { docker compose up -d }
# Restart the UPI app so its drift buffer starts empty and the 500-input windows line up.
docker compose restart app-upi | Out-Null

Write-Host "[2/4] Waiting for the UPI API and loading the model..." -ForegroundColor Cyan
Wait-Api "http://localhost:8006/"
# First request loads the model from MLflow (~3 s); do it now, then restart to drop it from
# the drift buffer.
Invoke-RestMethod -Method Post -ContentType application/json `
    -Body ((Get-Content frontend\presets_upi.json -Raw | ConvertFrom-Json).legit.values | ConvertTo-Json) `
    http://localhost:8006/prediction_api | Out-Null
docker compose restart app-upi | Out-Null
Wait-Api "http://localhost:8006/"
Invoke-WebRequest -UseBasicParsing http://localhost:8006/ui | Out-Null

Write-Host "[3/4] Sending 500 payments from NORMAL months (days 60-89)..." -ForegroundColor Cyan
& $py demo\send_upi_traffic.py --n 500
if ($LASTEXITCODE -ne 0) { throw "send_upi_traffic.py failed" }

Write-Host "      Waiting 45 s so Prometheus records the baseline point for Grafana..." -ForegroundColor Cyan
Start-Sleep 45

Write-Host "[4/4] Sending 500 payments from the FESTIVE SEASON (day 120+)..." -ForegroundColor Cyan
& $py demo\send_upi_traffic.py --n 500 --festive
if ($LASTEXITCODE -ne 0) { throw "send_upi_traffic.py failed" }

Write-Host "`nReady. Leave this stack running for the demo." -ForegroundColor Green
Write-Host "  UPI console : http://localhost:8006/ui"
Write-Host "  UPI Grafana : http://localhost:3000/d/upi-input-drift  (admin / admin, last 15 minutes)"
