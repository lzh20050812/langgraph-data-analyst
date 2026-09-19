param(
    [switch]$NoBrowser,
    [switch]$Operations
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dashboardUrl = "http://127.0.0.1:8000"
$openUrl = if ($Operations) { "$dashboardUrl/#/operations" } else { $dashboardUrl }
$healthUrl = "$dashboardUrl/health/live"
$stdoutPath = Join-Path $env:TEMP "ai-data-analyst-backend.stdout.log"
$stderrPath = Join-Path $env:TEMP "ai-data-analyst-backend.stderr.log"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project virtual environment not found: $pythonPath"
}

$running = $false
try {
    $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
    $running = $health.status -eq "ok"
} catch {
    $running = $false
}

if (-not $running) {
    Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    for ($attempt = 1; $attempt -le 20; $attempt++) {
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
            if ($health.status -eq "ok") {
                $running = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
}

if (-not $running) {
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath -Tail 30
    }
    throw "Backend startup failed. Review the log output above."
}

Write-Host "Platform started: $openUrl" -ForegroundColor Green
if (-not $NoBrowser) {
    Start-Process $openUrl
}
