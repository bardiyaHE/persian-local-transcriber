[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$RunDir,
    [int]$Port = 18080,
    [int]$TimeoutSeconds = 180,
    [int]$MaxRegions = 8
)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$ResolvedRun = (Resolve-Path -LiteralPath $RunDir -ErrorAction Stop).Path
$Python = Join-Path $Root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Local environment is missing. Run setup.ps1 first.' }
& (Join-Path $Root 'start_local_qwen.ps1') -Port $Port
& $Python (Join-Path $Root 'src/local_qwen_reranker_v10.py') `
    --run-dir $ResolvedRun `
    --medical-index (Join-Path $Root 'offline-lexicon/combined-medical-drug-index.json') `
    --server-url "http://127.0.0.1:$Port" `
    --timeout $TimeoutSeconds `
    --max-regions $MaxRegions
if ($LASTEXITCODE -ne 0) { throw "V10 local Qwen reranker failed with exit code $LASTEXITCODE" }
Write-Host "V10 result: $(Join-Path $ResolvedRun 'final-delivery/10-local-qwen-reranker/final-v10.txt')"
