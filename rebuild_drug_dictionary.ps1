param(
    [switch]$RefreshDownload,
    [switch]$Offline
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$SourceDir = Join-Path $Root 'offline-lexicon\persian-drug-names'
$Csv = Join-Path $SourceDir 'train.csv'
$BaseIndex = Join-Path $Root 'offline-lexicon\persianmedqa-medical-index.json'
$Output = Join-Path $Root 'offline-lexicon\combined-medical-drug-index.json'
$Revision = '9ca2bcf9af0dce18e9e7d3ce5942c26a2f4be811'
$Url = "https://huggingface.co/datasets/dadashzadeh/Collection-of-drug-names-in-Persian/resolve/$Revision/train.csv"
$ExpectedSha256 = '0C87A89EEA527C8F00C294C3605BB0A37DFC37B579FEF8BDA5002D59C7A858FC'

if (($RefreshDownload -or -not (Test-Path -LiteralPath $Csv)) -and $Offline) {
    throw "Offline dictionary build: pinned Persian drug CSV is missing or refresh was requested: $Csv"
}
if ($RefreshDownload -or -not (Test-Path -LiteralPath $Csv)) {
    New-Item -ItemType Directory -Force -Path $SourceDir | Out-Null
    Invoke-WebRequest -Uri $Url -OutFile $Csv
}

$ActualSha256 = (Get-FileHash -LiteralPath $Csv -Algorithm SHA256).Hash
if ($ActualSha256 -ne $ExpectedSha256) {
    throw "Persian drug CSV checksum mismatch. Expected $ExpectedSha256, got $ActualSha256"
}

& $Python (Join-Path $Root 'src\build_combined_drug_index.py') `
    --base-index $BaseIndex `
    --persian-drug-csv $Csv `
    --output $Output

if ($LASTEXITCODE -ne 0) {
    throw "Drug dictionary build failed with exit code $LASTEXITCODE"
}
