# Download LongMemEval-S cleaned dataset (~265MB)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Out = Join-Path $Root "data\longmemeval_s_cleaned.json"
New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
$Uri = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
Write-Host "Downloading to $Out ..."
Invoke-WebRequest -Uri $Uri -OutFile $Out
Get-Item $Out | Select-Object FullName, @{N='SizeMB';E={[math]::Round($_.Length/1MB,2)}}
