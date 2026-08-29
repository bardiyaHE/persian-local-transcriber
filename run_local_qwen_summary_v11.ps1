[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$RunDir,
    [int]$Port = 18080,
    [int]$TimeoutSeconds = 180,
    [int]$GoogleTimeoutSeconds = 4,
    [string]$SourceTranscript,
    [switch]$DisableGoogleDrugCorrection,
    [switch]$RevalidateExisting
)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$ResolvedRun = (Resolve-Path -LiteralPath $RunDir -ErrorAction Stop).Path
$Python = Join-Path $Root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Local environment is missing. Run setup.ps1 first.' }
$env:PYTHONIOENCODING = 'utf-8'
$SummaryJson = Join-Path $ResolvedRun 'final-delivery/11-local-qwen-summary/final-summary-v11.json'
for ($Attempt = 1; $Attempt -le 2; $Attempt++) {
    if (-not $RevalidateExisting) {
        & (Join-Path $Root 'start_local_qwen.ps1') -Port $Port
    }
    $SummaryArgs = @(
        (Join-Path $Root 'src/local_qwen_summarizer_v11.py'),
        '--run-dir', $ResolvedRun,
        '--medical-index', (Join-Path $Root 'offline-lexicon/combined-medical-drug-index.json'),
        '--server-url', "http://127.0.0.1:$Port",
        '--timeout', [string]$TimeoutSeconds,
        '--google-timeout', [string]$GoogleTimeoutSeconds,
        '--google-cache', (Join-Path $Root 'runtime/google-drug-spelling-cache.json')
    )
    if (-not $DisableGoogleDrugCorrection) { $SummaryArgs += '--google-drug-correction' }
    if ($SourceTranscript) {
        $ResolvedSource = (Resolve-Path -LiteralPath $SourceTranscript -ErrorAction Stop).Path
        $SummaryArgs += @('--source-transcript', $ResolvedSource)
    }
    if ($RevalidateExisting) { $SummaryArgs += '--revalidate-existing' }
    & $Python @SummaryArgs
    if ($LASTEXITCODE -ne 0) {
        throw "V11 local Qwen summarizer failed with exit code $LASTEXITCODE"
    }
    $FallbackReason = ''
    if (Test-Path -LiteralPath $SummaryJson) {
        $Payload = Get-Content -LiteralPath $SummaryJson -Raw | ConvertFrom-Json
        $FallbackReason = [string]$Payload.fallback_reason
    }
    $Retryable = $FallbackReason -match 'ConnectionResetError|ConnectionRefusedError|URLError|TimeoutError|timed out|forcibly closed'
    if (-not $Retryable -or $Attempt -eq 2) { break }
    Write-Warning "Local Qwen connection failed; restarting and retrying once. Reason: $FallbackReason"
    Start-Sleep -Seconds 1
}
Write-Host "V11 summary: $(Join-Path $ResolvedRun 'final-delivery/11-local-qwen-summary/final-summary-v11.txt')"
