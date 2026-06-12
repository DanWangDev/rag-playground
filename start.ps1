# RAG Playground — Windows PowerShell Launcher
# Usage: .\start.ps1
# One-click: right-click → "Run with PowerShell"

param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

function Write-Step { param($Msg) Write-Host "`n  $Msg" -ForegroundColor Cyan }
function Write-OK { param($Msg) Write-Host "  ✅ $Msg" -ForegroundColor Green }
function Write-Warn { param($Msg) Write-Host "  ⚠️  $Msg" -ForegroundColor Yellow }
function Write-Err { param($Msg) Write-Host "  ❌ $Msg" -ForegroundColor Red }

# ── Stop ──
if ($Stop) {
    Write-Step "Stopping RAG Playground..."
    Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*uvicorn*server.main*" } | Stop-Process -Force
    Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*vite*" } | Stop-Process -Force
    Write-OK "All services stopped"
    exit 0
}

# ── Status ──
if ($Status) {
    Write-Host "`n📊 RAG Playground Status" -ForegroundColor Cyan
    Write-Host ""

    $ollamaOk = try { Invoke-WebRequest -Uri "http://localhost:11434/" -UseBasicParsing -TimeoutSec 2; $true } catch { $false }
    $backendOk = try { (Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch { $false }

    Write-Host "  Ollama:    $(if ($ollamaOk) { '✅ running' } else { '❌ not running' })"
    Write-Host "  Backend:   $(if ($backendOk) { '✅ running' } else { '❌ not running' })"
    Write-Host "  Frontend:  check http://localhost:5173 in browser"
    exit 0
}

# ── Main: Start everything ──

Write-Host "`n🚀 RAG Playground Launcher" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# Step 1: Check Ollama
Write-Step "Step 1/4: Checking Ollama..."
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434/" -UseBasicParsing -TimeoutSec 3
    Write-OK "Ollama is running"
} catch {
    Write-Err "Ollama is not running on localhost:11434"
    Write-Host "  Start Ollama first: ollama serve"
    Write-Host "  Then run this script again."
    exit 1
}

# Step 2: Check models
Write-Step "Step 2/4: Checking models..."
python scripts/check_ollama.py
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Some models are missing. Pulling them now..."
    python scripts/pull_models.py
}

if ($CheckOnly) {
    Write-OK "All checks passed!"
    exit 0
}

# Step 3: Install deps if needed
Write-Step "Step 3/4: Checking dependencies..."
try {
    python -c "import fastapi" 2>$null
} catch {
    Write-Warn "Installing Python dependencies..."
    pip install -e ".[dev]"
}
if (-not (Test-Path "frontend/node_modules")) {
    Write-Warn "Installing frontend dependencies..."
    Set-Location frontend; npm install; Set-Location ..
}
Write-OK "Dependencies OK"

# Step 4: Launch services
Write-Step "Step 4/4: Starting services..."

$backendJob = Start-Job -Name "rag-backend" -ScriptBlock {
    Set-Location $using:PWD
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | Out-Null
}
Write-OK "Backend starting on http://localhost:8000"

$frontendJob = Start-Job -Name "rag-frontend" -ScriptBlock {
    Set-Location "$using:PWD/frontend"
    npm run dev 2>&1 | Out-Null
}
Write-OK "Frontend starting on http://localhost:5173"

# Wait for services
Start-Sleep -Seconds 3

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  RAG Playground is running!" -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend:  http://localhost:5173"
Write-Host "  Backend:   http://localhost:8000"
Write-Host "  API Docs:  http://localhost:8000/docs"
Write-Host "  Health:    http://localhost:8000/api/health"
Write-Host ""
Write-Host "  Run .\start.ps1 -Stop to shut down"
Write-Host "  Run .\start.ps1 -Status to check"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# Open browser
Start-Process "http://localhost:5173"

Write-Host "`nPress Ctrl+C to stop all services...`n"
try {
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    Write-Step "Shutting down..."
    $backendJob | Stop-Job -PassThru | Remove-Job
    $frontendJob | Stop-Job -PassThru | Remove-Job
    Write-OK "All services stopped"
}
