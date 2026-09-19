param(
    [switch]$SkipBuild,
    [ValidateRange(30, 600)]
    [int]$HealthTimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $projectRoot "docker\docker-compose.yml"
$projectName = "ai_analytics"

$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $dockerPath = $dockerCommand.Source
} else {
    $dockerPath = Join-Path $env:LOCALAPPDATA "Programs\Docker\Docker Desktop\resources\bin\docker.exe"
}

if (-not (Test-Path -LiteralPath $dockerPath)) {
    throw "Docker CLI not found. Install or start Docker Desktop first."
}

function Invoke-Docker {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    & $dockerPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed: docker $($Arguments -join ' ')"
    }
}

function Get-ContainerHealth {
    param([string]$Container)

    $value = & $dockerPath inspect --format `
        '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' `
        $Container 2>$null
    if ($LASTEXITCODE -ne 0) { return "missing" }
    return ($value | Select-Object -First 1).Trim()
}

Push-Location $projectRoot
try {
    Invoke-Docker -Arguments @("compose", "-p", $projectName, "-f", $composeFile, "config", "--quiet")

    if (-not $SkipBuild) {
        Invoke-Docker -Arguments @("compose", "-p", $projectName, "-f", $composeFile, "build", "app")
    }

    Invoke-Docker -Arguments @("compose", "-p", $projectName, "-f", $composeFile, "up", "-d", "--no-build")

    $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
    do {
        $appHealth = Get-ContainerHealth "ai_analytics_app"
        $workerHealth = Get-ContainerHealth "ai_analytics_worker"
        $mysqlHealth = Get-ContainerHealth "ai_analytics_mysql"
        if ($appHealth -eq "healthy" -and
            $workerHealth -eq "healthy" -and
            $mysqlHealth -eq "healthy") {
            break
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    if ($appHealth -ne "healthy" -or
        $workerHealth -ne "healthy" -or
        $mysqlHealth -ne "healthy") {
        Invoke-Docker -Arguments @("compose", "-p", $projectName, "-f", $composeFile, "ps", "-a")
        throw "Containers did not become healthy: app=$appHealth worker=$workerHealth mysql=$mysqlHealth"
    }

    $dbInitExitCode = (& $dockerPath inspect --format '{{.State.ExitCode}}' ai_analytics_db_init).Trim()
    if ($LASTEXITCODE -ne 0 -or $dbInitExitCode -ne "0") {
        throw "Database initialization did not exit successfully: exit=$dbInitExitCode"
    }

    $workerRestarts = [int](& $dockerPath inspect --format '{{.RestartCount}}' ai_analytics_worker)
    if ($LASTEXITCODE -ne 0 -or $workerRestarts -ne 0) {
        throw "Worker restart count is not zero: $workerRestarts"
    }

    & $dockerPath exec ai_analytics_app python -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Container dependency consistency check failed."
    }

    $appEnvFile = (& $dockerPath exec ai_analytics_app python -c `
        "from pathlib import Path; print(str(Path('/app/.env').exists()).lower())").Trim()
    $workerEnvFile = (& $dockerPath exec ai_analytics_worker python -c `
        "from pathlib import Path; print(str(Path('/app/.env').exists()).lower())").Trim()
    if ($LASTEXITCODE -ne 0 -or $appEnvFile -ne "false" -or $workerEnvFile -ne "false") {
        throw "A container exposes /app/.env: app=$appEnvFile worker=$workerEnvFile"
    }

    $baseUrl = "http://127.0.0.1:8000"
    $live = Invoke-RestMethod -Uri "$baseUrl/health/live" -TimeoutSec 10
    $ready = Invoke-RestMethod -Uri "$baseUrl/health/ready" -TimeoutSec 10
    $frontend = Invoke-WebRequest -Uri "$baseUrl/" -UseBasicParsing -TimeoutSec 10

    if ($live.status -ne "ok" -or
        $ready.status -ne "ok" -or
        -not $ready.mysql -or
        -not $ready.worker -or
        $frontend.StatusCode -ne 200) {
        throw "HTTP smoke check failed."
    }

    Invoke-Docker -Arguments @("compose", "-p", $projectName, "-f", $composeFile, "ps", "-a")
    [ordered]@{
        success = $true
        app = $appHealth
        worker = $workerHealth
        mysql = $mysqlHealth
        db_init_exit = [int]$dbInitExitCode
        worker_restarts = $workerRestarts
        live = $live.status
        ready = $ready.status
        frontend_status = $frontend.StatusCode
        dependencies = "consistent"
        dotenv_file_present = $false
        build_skipped = [bool]$SkipBuild
    } | ConvertTo-Json
} finally {
    Pop-Location
}

Write-Host "Docker release verification passed." -ForegroundColor Green
