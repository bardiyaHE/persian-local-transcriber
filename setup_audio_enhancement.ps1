[CmdletBinding()]
param([switch]$Offline)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$ConfigPath = Join-Path $Root 'runtime\pyannote-enhancer.json'
$Config = if (Test-Path -LiteralPath $ConfigPath) {
    Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
} else { $null }

function Resolve-ProjectPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Value))
}

$DemucsPython = Resolve-ProjectPath $(if ($Config.demucs_python) {
    [string]$Config.demucs_python
} else { 'runtime\demucs-venv\Scripts\python.exe' })
$PyannotePython = Resolve-ProjectPath $(if ($Config.pyannote_python) {
    [string]$Config.pyannote_python
} else { 'runtime\pyannote-venv\Scripts\python.exe' })
$DemucsCache = Resolve-ProjectPath $(if ($Config.demucs_cache) {
    [string]$Config.demucs_cache
} else { 'runtime\demucs-cache' })
$PyannoteCache = Resolve-ProjectPath $(if ($Config.pyannote_cache) {
    [string]$Config.pyannote_cache
} else { 'runtime\pyannote-cache' })

$SystemPython = (Get-Command python -ErrorAction Stop).Source
foreach ($Entry in @(
    @{ Python=$DemucsPython; Requirements='requirements-demucs.txt'; Import='demucs, torch' },
    @{ Python=$PyannotePython; Requirements='requirements-pyannote.txt'; Import='pyannote.audio, torch' }
)) {
    if (-not (Test-Path -LiteralPath $Entry.Python)) {
        if ($Offline) { throw "Offline setup: enhancement environment is missing: $($Entry.Python)" }
        $VenvRoot = Split-Path -Parent (Split-Path -Parent $Entry.Python)
        & $SystemPython -m venv $VenvRoot
        if ($LASTEXITCODE -ne 0) { throw "Could not create enhancement environment: $VenvRoot" }
    }
    & $Entry.Python -c "import $($Entry.Import)"
    if ($LASTEXITCODE -ne 0) {
        if ($Offline) { throw "Offline setup: packages are missing in $($Entry.Python)" }
        & $Entry.Python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed in $($Entry.Python)" }
        & $Entry.Python -m pip install --requirement (Join-Path $Root $Entry.Requirements)
        if ($LASTEXITCODE -ne 0) { throw "Enhancement dependency installation failed: $($Entry.Requirements)" }
    }
}

New-Item -ItemType Directory -Force -Path $DemucsCache,$PyannoteCache | Out-Null
$OriginalHfHome = $env:HF_HOME
$env:HF_HOME = $DemucsCache
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYANNOTE_METRICS_ENABLED = '0'
$env:DO_NOT_TRACK = '1'
if ($Offline) { $env:HF_HUB_OFFLINE = '1' }

$DemucsRevision = 'cbc8a9b1a87023b7fd74e7b3412e6321c0eab003'
& $DemucsPython -c "from huggingface_hub import hf_hub_download; [hf_hub_download('adefossez/HTDemucs', f, revision='$DemucsRevision') for f in ('955717e8.safetensors','htdemucs.yaml')]"
if ($LASTEXITCODE -ne 0) { throw 'Pinned HTDemucs model caching failed.' }
if ($null -eq $OriginalHfHome) {
    Remove-Item Env:HF_HOME -ErrorAction SilentlyContinue
} else {
    $env:HF_HOME = $OriginalHfHome
}

$PyannoteArgs = @(
    (Join-Path $Root 'src\cache_pyannote_model.py'),
    '--cache-dir', $PyannoteCache
)
if ($Offline) { $PyannoteArgs += '--offline' }
& $PyannotePython @PyannoteArgs
if ($LASTEXITCODE -ne 0) {
    throw 'pyannote Community-1 caching failed. Accept the model terms and set HF_TOKEN.'
}

[ordered]@{
    schema_version = 1
    demucs = [ordered]@{
        package = '4.1.0'; model = 'htdemucs'; revision = $DemucsRevision
        python = $DemucsPython; cache = $DemucsCache
    }
    pyannote = [ordered]@{
        package = '4.0.7'; model = 'pyannote/speaker-diarization-community-1'
        python = $PyannotePython; cache = $PyannoteCache
    }
    runtime_network_disabled = $true
    audio_uploaded = $false
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 `
    (Join-Path $Root 'runtime\audio-enhancement-manifest.json')

Write-Host 'Demucs + pyannote enhancement is installed and cached.'
