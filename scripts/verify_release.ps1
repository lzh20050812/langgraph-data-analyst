param(
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project virtual environment not found: $pythonPath"
}

Push-Location $projectRoot
try {
    & $pythonPath -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Python dependency consistency check failed" }

    & $pythonPath -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Python regression failed" }

    & $pythonPath -m evaluation.experiments.stage5_deterministic_regression --check
    if ($LASTEXITCODE -ne 0) { throw "Deterministic evaluation audit failed" }

    & $pythonPath -m compileall -q agents api config
    if ($LASTEXITCODE -ne 0) { throw "Python compile check failed" }

    if (-not $SkipFrontend) {
        Push-Location (Join-Path $projectRoot "frontend")
        try {
            & npm run typecheck
            if ($LASTEXITCODE -ne 0) { throw "Frontend typecheck failed" }
            & npm run build
            if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
        } finally {
            Pop-Location
        }
    }
} finally {
    Pop-Location
}

Write-Host "Release verification passed." -ForegroundColor Green
