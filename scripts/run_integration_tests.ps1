# Start Docker services and run Redis + pgvector integration tests.
# Prerequisites: Docker Desktop running, venv activated, mnemo[redis,pgvector] installed.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> Starting redis + postgres (pgvector) via docker compose..."
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Error "docker compose failed. Is Docker Desktop running?"
}

Write-Host "==> Waiting for healthchecks..."
Start-Sleep -Seconds 8

$env:MNEMO_REDIS_URL = "redis://localhost:6379/0"
$env:MNEMO_PGVECTOR_DSN = "postgresql://mnemo:mnemo@localhost:5433/mnemo"

Write-Host "==> Redis test..."
pytest tests/test_redis_backend.py -v
Write-Host "==> Pgvector test..."
pytest tests/test_pgvector_backend.py -v

Write-Host "Done."
