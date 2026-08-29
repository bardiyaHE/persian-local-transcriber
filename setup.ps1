[CmdletBinding()]
param(
    [ValidateSet('Lite', 'Full')][string]$Profile = 'Full',
    [switch]$Offline,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$ProfileKey = $Profile.ToLowerInvariant()
$Directories = @(
    'models/medium', 'models/large-v3-turbo', 'models/large-v3',
    'models/qwen3.5-35b-a3b-gguf', 'runtime/ffmpeg', 'runtime/deepfilternet',
    'runtime/cuda-libs', 'runtime/llama.cpp/cpu', 'runtime/llama.cpp/cuda',
    'wheelhouse', 'offline-lexicon', 'offline-corpus', 'inputs', 'outputs', 'src'
)
foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $Directory) | Out-Null
}

$DownloadCache = if ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA 'PersianLocalTranscriber\downloads'
} else {
    Join-Path ([System.IO.Path]::GetTempPath()) 'PersianLocalTranscriber-downloads'
}
New-Item -ItemType Directory -Force -Path $DownloadCache | Out-Null

function Invoke-ResumableDownload {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Destination
    )
    if ($Offline) {
        if (-not (Test-Path -LiteralPath $Destination)) {
            throw "Offline setup: cached download is missing: $Destination"
        }
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    $Curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($Curl) {
        & $Curl.Source --fail --location --retry 4 --retry-delay 2 `
            --continue-at - --output $Destination $Url
        if ($LASTEXITCODE -ne 0) {
            Remove-Item -LiteralPath $Destination -Force -ErrorAction SilentlyContinue
            & $Curl.Source --fail --location --retry 4 --retry-delay 2 `
                --output $Destination $Url
        }
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $Destination
    }
}

function Install-VerifiedArchive {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Sha256,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][string]$ArchiveName
    )
    $Archive = Join-Path $DownloadCache $ArchiveName
    Invoke-ResumableDownload -Url $Url -Destination $Archive
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash
    if ($Actual -ne $Sha256) {
        Remove-Item -LiteralPath $Archive -Force
        throw "Archive SHA-256 mismatch for $ArchiveName"
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
    Remove-Item -LiteralPath $Archive -Force
}

function Install-PublicLexicon {
    param([switch]$CheckOnly)
    $ResourceRoot = Join-Path $Root 'resources\lexicon'
    $ManifestPath = Join-Path $ResourceRoot 'manifest.json'
    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        throw "Bundled lexicon manifest is missing: $ManifestPath"
    }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    foreach ($Entry in $Manifest.files) {
        $Source = Join-Path $ResourceRoot ([string]$Entry.name)
        if (-not (Test-Path -LiteralPath $Source)) {
            throw "Bundled lexicon resource is missing: $Source"
        }
        $ActualBytes = (Get-Item -LiteralPath $Source).Length
        $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash
        if ($ActualBytes -ne [int64]$Entry.bytes -or $ActualHash -ne [string]$Entry.sha256) {
            throw "Bundled lexicon resource failed validation: $($Entry.name)"
        }
        $DestinationName = if ($Entry.name -eq 'SOURCES.md') {
            'LEXICON_SOURCES.md'
        } else {
            [string]$Entry.name
        }
        if (-not $CheckOnly) {
            Copy-Item -LiteralPath $Source -Destination `
                (Join-Path $Root ('offline-lexicon\' + $DestinationName)) -Force
        }
    }
}

$Threads = [Environment]::ProcessorCount
$Nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$Device = if ($Nvidia) { 'cuda' } else { 'cpu' }
$ComputeType = if ($Nvidia) { 'float16' } else { 'int8' }
Write-Host "Profile=$ProfileKey; logical CPU threads=$Threads; device=$Device compute_type=$ComputeType"

if ($ValidateOnly) {
    if ($ProfileKey -eq 'full') { Install-PublicLexicon -CheckOnly }
    [string[]]$ExpectedWhisperModels = if ($ProfileKey -eq 'lite') {
        @('large-v3-turbo')
    } else {
        @('medium', 'large-v3-turbo', 'large-v3')
    }
    [ordered]@{
        status = 'ok'
        profile = $ProfileKey
        downloads_started = $false
        bundled_lexicon_validated = ($ProfileKey -eq 'full')
        expected_whisper_models = $ExpectedWhisperModels
        local_qwen_required = ($ProfileKey -eq 'full')
        local_ngram_database_built_during_setup = ($ProfileKey -eq 'full')
    } | ConvertTo-Json -Depth 4 | Write-Host
    return
}

$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
    & $Python -m venv (Join-Path $Root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating the local Python environment failed.' }
}
$PythonVersion = (& $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($PythonVersion -ne '3.12') {
    throw "Python 3.12 is required; the local environment uses Python $PythonVersion."
}
if (-not $Offline) {
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }
}

$Wheelhouse = Join-Path $Root 'wheelhouse'
if (-not $Offline) {
    & $VenvPython -m pip download --dest $Wheelhouse `
        --requirement (Join-Path $Root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Downloading the Python wheelhouse failed.' }
}
& $VenvPython -m pip install --no-index --find-links $Wheelhouse `
    --requirement (Join-Path $Root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installing from the local wheelhouse failed.' }

if ($ProfileKey -eq 'full') {
    $QwenRevision = 'bc014a17be43adabd7066b7a86075ff935c6a4e2'
    $QwenFileName = 'Qwen3.5-35B-A3B-UD-Q4_K_L.gguf'
    $QwenModelDir = Join-Path $Root 'models\qwen3.5-35b-a3b-gguf'
    $QwenModel = Join-Path $QwenModelDir $QwenFileName
    $QwenSha256 = 'B65BBA850B65E989AC8C37978970B2FE2ED3AA404FCB8408C9B3F2DF13E6AB0B'
    if (-not (Test-Path -LiteralPath $QwenModel)) {
        if ($Offline) { throw "Offline setup: local Qwen model is missing: $QwenModel" }
        $Hf = Join-Path $Root '.venv\Scripts\hf.exe'
        if (-not (Test-Path -LiteralPath $Hf)) {
            throw 'Hugging Face CLI is missing from the local environment.'
        }
        $QwenStage = Join-Path $DownloadCache 'qwen3.5-35b-a3b'
        New-Item -ItemType Directory -Force -Path $QwenStage | Out-Null
        & $Hf download 'unsloth/Qwen3.5-35B-A3B-GGUF' $QwenFileName `
            --revision $QwenRevision --local-dir $QwenStage
        if ($LASTEXITCODE -ne 0) { throw 'Pinned local Qwen download failed.' }
        $StagedQwen = Join-Path $QwenStage $QwenFileName
        $StagedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedQwen).Hash
        if ($StagedHash -ne $QwenSha256) {
            throw "Staged Qwen SHA-256 mismatch: $StagedHash"
        }
        Move-Item -LiteralPath $StagedQwen -Destination $QwenModel -Force
    }
    $ActualQwenHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $QwenModel).Hash
    if ($ActualQwenHash -ne $QwenSha256) {
        throw "Local Qwen SHA-256 mismatch: $ActualQwenHash"
    }

    $LlamaRelease = 'b10642'
    $LlamaBaseUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaRelease"
    $LlamaRuntimeName = if ($Nvidia) { 'cuda' } else { 'cpu' }
    $LlamaRuntime = Join-Path $Root "runtime\llama.cpp\$LlamaRuntimeName"
    $LlamaServer = Join-Path $LlamaRuntime 'llama-server.exe'
    if (-not (Test-Path -LiteralPath $LlamaServer)) {
        if ($Offline) { throw "Offline setup: llama.cpp runtime is missing: $LlamaServer" }
        if ($Nvidia) {
            Install-VerifiedArchive `
                -Url "$LlamaBaseUrl/llama-b10642-bin-win-cuda-12.4-x64.zip" `
                -Sha256 '03DE5617661F874BB8823E0938D9170094D17F857965B501AA687B340FFF9222' `
                -Destination $LlamaRuntime `
                -ArchiveName 'llama-b10642-bin-win-cuda-12.4-x64.zip'
            Install-VerifiedArchive `
                -Url "$LlamaBaseUrl/cudart-llama-bin-win-cuda-12.4-x64.zip" `
                -Sha256 '8C79A9B226DE4B3CACFD1F83D24F962D0773BE79F1E7B75C6AF4DED7E32AE1D6' `
                -Destination $LlamaRuntime `
                -ArchiveName 'cudart-llama-bin-win-cuda-12.4-x64.zip'
        } else {
            Install-VerifiedArchive `
                -Url "$LlamaBaseUrl/llama-b10642-bin-win-cpu-x64.zip" `
                -Sha256 'B90C4B018DE11961A25A2555427FA1576267E6499B3E2F873433D9188EC929E2' `
                -Destination $LlamaRuntime `
                -ArchiveName 'llama-b10642-bin-win-cpu-x64.zip'
        }
    }
    & $LlamaServer --version
    if ($LASTEXITCODE -ne 0) { throw 'Pinned llama.cpp runtime validation failed.' }
}

if ($Nvidia) {
    if (-not $Offline) {
        & $VenvPython -m pip download --dest $Wheelhouse nvidia-cublas-cu12 nvidia-cudnn-cu12
        if ($LASTEXITCODE -ne 0) { throw 'CUDA 12/cuDNN 9 library download failed.' }
    }
    & $VenvPython -m pip install --no-index --find-links $Wheelhouse `
        --target (Join-Path $Root 'runtime\cuda-libs') nvidia-cublas-cu12 nvidia-cudnn-cu12
    if ($LASTEXITCODE -ne 0) { throw 'CUDA 12/cuDNN 9 local install failed.' }
}

$Ffmpeg = Join-Path $Root 'runtime\ffmpeg\ffmpeg.exe'
if (-not (Test-Path -LiteralPath $Ffmpeg)) {
    $FfmpegArchive = Join-Path $DownloadCache 'ffmpeg-master-latest-win64-gpl.zip'
    Invoke-ResumableDownload `
        -Url 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' `
        -Destination $FfmpegArchive
    $Extract = Join-Path $DownloadCache 'ffmpeg-extracted'
    if (Test-Path -LiteralPath $Extract) { Remove-Item -LiteralPath $Extract -Recurse -Force }
    Expand-Archive -LiteralPath $FfmpegArchive -DestinationPath $Extract -Force
    $Bin = Get-ChildItem -LiteralPath $Extract -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    if (-not $Bin) { throw 'Downloaded FFmpeg archive did not contain ffmpeg.exe.' }
    Copy-Item -LiteralPath (Join-Path $Bin.DirectoryName 'ffmpeg.exe') -Destination $Ffmpeg -Force
    Copy-Item -LiteralPath (Join-Path $Bin.DirectoryName 'ffprobe.exe') `
        -Destination (Join-Path $Root 'runtime\ffmpeg\ffprobe.exe') -Force
    Remove-Item -LiteralPath $FfmpegArchive -Force
    Remove-Item -LiteralPath $Extract -Recurse -Force
}
& $Ffmpeg -version | Select-Object -First 1
if ($LASTEXITCODE -ne 0) { throw 'Local FFmpeg validation failed.' }

$DeepFilter = Join-Path $Root 'runtime\deepfilternet\deep-filter.exe'
if (-not (Test-Path -LiteralPath $DeepFilter)) {
    if ($Offline) { throw "Offline setup: DeepFilterNet is missing: $DeepFilter" }
    $DfRelease = Invoke-RestMethod `
        -Uri 'https://api.github.com/repos/Rikorose/DeepFilterNet/releases/latest' `
        -Headers @{ 'User-Agent'='whisper-persian-local' }
    $DfAsset = $DfRelease.assets | Where-Object {
        $_.name -match 'x86_64-pc-windows-msvc\.exe$'
    } | Select-Object -First 1
    if (-not $DfAsset) { throw 'Could not locate the DeepFilterNet Windows x64 binary.' }
    $DeepFilterCache = Join-Path $DownloadCache ([string]$DfAsset.name)
    Invoke-ResumableDownload -Url $DfAsset.browser_download_url -Destination $DeepFilterCache
    Copy-Item -LiteralPath $DeepFilterCache -Destination $DeepFilter -Force
    $DfRelease | Select-Object tag_name,published_at,html_url | ConvertTo-Json |
        Set-Content -Encoding utf8 (Join-Path $Root 'runtime\deepfilternet\release.json')
}
& $DeepFilter --version
if ($LASTEXITCODE -ne 0) { throw 'DeepFilterNet validation failed.' }

$ModelArgs = @('--root', $Root, '--profile', $ProfileKey)
if ($Offline) { $ModelArgs += '--local-only' }
& $VenvPython (Join-Path $Root 'src\download_models.py') @ModelArgs
if ($LASTEXITCODE -ne 0) { throw 'CTranslate2 model download or validation failed.' }

if ($ProfileKey -eq 'full') {
    $EncoderArgs = @('--root', $Root)
    if ($Offline) { $EncoderArgs += '--local-only' }
    & $VenvPython (Join-Path $Root 'src\download_semantic_encoder.py') @EncoderArgs
    if ($LASTEXITCODE -ne 0) { throw 'Semantic encoder download or validation failed.' }

    Install-PublicLexicon

    $CorpusArgs = @(
        '--output-dir', (Join-Path $Root 'offline-corpus'),
        '--public-only'
    )
    if ($Offline) { $CorpusArgs += '--local-only' }
    & $VenvPython (Join-Path $Root 'src\build_domain_corpus.py') @CorpusArgs
    if ($LASTEXITCODE -ne 0) { throw 'Local public-source n-gram database build failed.' }
}

$PackageVersions = & $VenvPython -c `
    "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ['faster-whisper','ctranslate2','huggingface-hub','numpy','gradio']}))"
$InstallManifest = [ordered]@{
    schema_version = 1
    created_at = (Get-Date).ToUniversalTime().ToString('o')
    profile = $ProfileKey
    cpu_threads = $Threads
    nvidia = [bool]$Nvidia
    device = $Device
    compute_type = $ComputeType
    packages = ($PackageVersions | ConvertFrom-Json)
    external_speech_fallback_enabled_by_default = $false
    bundled_user_data = $false
}
$InstallManifest | ConvertTo-Json -Depth 5 |
    Set-Content -Encoding utf8 (Join-Path $Root 'runtime\install-profile.json')

& $VenvPython (Join-Path $Root 'src\healthcheck.py') `
    --root $Root --profile $ProfileKey
if ($LASTEXITCODE -ne 0) { throw 'Installation health check failed.' }

& $VenvPython (Join-Path $Root 'src\web_app.py') --smoke-test
if ($LASTEXITCODE -ne 0) { throw 'Web interface smoke test failed.' }

Write-Host "Setup completed successfully. Profile=$ProfileKey"
Write-Host 'Start the local interface with: .\launch_ui.ps1'
