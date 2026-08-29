[CmdletBinding()]
param([switch]$Offline)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Dirs = @(
    'models/medium','models/large-v3-turbo','models/large-v3','models/qwen3.5-35b-a3b-gguf',
    'runtime/ffmpeg','runtime/deepfilternet','runtime/cuda-libs','runtime/llama.cpp/cpu',
    'runtime/llama.cpp/cuda','wheelhouse','offline-lexicon','offline-corpus','inputs','outputs','src'
)
foreach ($Dir in $Dirs) { New-Item -ItemType Directory -Force -Path (Join-Path $Root $Dir) | Out-Null }

$Threads = [Environment]::ProcessorCount
$Nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$Device = if ($Nvidia) { 'cuda' } else { 'cpu' }
$ComputeType = if ($Nvidia) { 'float16' } else { 'int8' }
Write-Host "Detected $Threads logical CPU threads; device=$Device compute_type=$ComputeType"

$VenvPython = Join-Path $Root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
    & $Python -m venv (Join-Path $Root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating local Python virtual environment failed.' }
}
if (-not $Offline) {
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }
}

$Wheelhouse = Join-Path $Root 'wheelhouse'
if (-not $Offline) {
    & $VenvPython -m pip download --dest $Wheelhouse --requirement (Join-Path $Root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Downloading Python wheelhouse failed.' }
}
& $VenvPython -m pip install --no-index --find-links $Wheelhouse --requirement (Join-Path $Root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installing from local wheelhouse failed.' }

$QwenRevision = 'bc014a17be43adabd7066b7a86075ff935c6a4e2'
$QwenFileName = 'Qwen3.5-35B-A3B-UD-Q4_K_L.gguf'
$QwenModelDir = Join-Path $Root 'models/qwen3.5-35b-a3b-gguf'
$QwenModel = Join-Path $QwenModelDir $QwenFileName
$QwenSha256 = 'B65BBA850B65E989AC8C37978970B2FE2ED3AA404FCB8408C9B3F2DF13E6AB0B'
if (-not (Test-Path -LiteralPath $QwenModel)) {
    if ($Offline) { throw "Offline setup: local Qwen model missing at $QwenModel" }
    $Hf = Join-Path $Root '.venv/Scripts/hf.exe'
    if (-not (Test-Path -LiteralPath $Hf)) { throw 'Hugging Face CLI is missing from the local environment.' }
    # Hugging Face uses a long temporary filename. Stage under the system temp
    # directory so deeply nested project paths do not exceed Windows path limits.
    $QwenStage = Join-Path ([System.IO.Path]::GetTempPath()) ("whisper-persian-qwen35-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $QwenStage | Out-Null
    try {
        & $Hf download 'unsloth/Qwen3.5-35B-A3B-GGUF' $QwenFileName `
            --revision $QwenRevision --local-dir $QwenStage
        if ($LASTEXITCODE -ne 0) { throw 'Pinned local Qwen GGUF download failed.' }
        $StagedQwen = Join-Path $QwenStage $QwenFileName
        $StagedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedQwen).Hash
        if ($StagedHash -ne $QwenSha256) {
            throw "Staged local Qwen SHA256 mismatch. Expected $QwenSha256 but got $StagedHash"
        }
        Move-Item -LiteralPath $StagedQwen -Destination $QwenModel
    } finally {
        if (Test-Path -LiteralPath $QwenStage) { Remove-Item -LiteralPath $QwenStage -Recurse -Force }
    }
}
$ActualQwenHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $QwenModel).Hash
if ($ActualQwenHash -ne $QwenSha256) {
    throw "Local Qwen SHA256 mismatch. Expected $QwenSha256 but got $ActualQwenHash"
}

function Install-VerifiedArchive {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Sha256,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][string]$ArchiveName
    )
    $Archive = Join-Path (Join-Path $Root 'runtime/llama.cpp') $ArchiveName
    Invoke-WebRequest -Uri $Url -OutFile $Archive
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash
    if ($Actual -ne $Sha256) {
        Remove-Item -LiteralPath $Archive -Force
        throw "llama.cpp archive SHA256 mismatch for $ArchiveName"
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
    Remove-Item -LiteralPath $Archive -Force
}

$LlamaRelease = 'b10642'
$LlamaBaseUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaRelease"
$LlamaRuntimeName = if ($Nvidia) { 'cuda' } else { 'cpu' }
$LlamaRuntime = Join-Path $Root "runtime/llama.cpp/$LlamaRuntimeName"
$LlamaServer = Join-Path $LlamaRuntime 'llama-server.exe'
if (-not (Test-Path -LiteralPath $LlamaServer)) {
    if ($Offline) { throw "Offline setup: llama.cpp $LlamaRuntimeName runtime missing at $LlamaServer" }
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
if ($LASTEXITCODE -ne 0) { throw 'Pinned local llama.cpp runtime validation failed.' }

if ($Nvidia) {
    if (-not $Offline) {
        & $VenvPython -m pip download --dest $Wheelhouse nvidia-cublas-cu12 nvidia-cudnn-cu12
        if ($LASTEXITCODE -ne 0) { throw 'NVIDIA CUDA 12/cuDNN 9 library download failed; refusing CPU fallback.' }
    }
    & $VenvPython -m pip install --no-index --find-links $Wheelhouse --target (Join-Path $Root 'runtime/cuda-libs') nvidia-cublas-cu12 nvidia-cudnn-cu12
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA CUDA 12/cuDNN 9 local install failed; refusing CPU fallback.' }
}

$Ffmpeg = Join-Path $Root 'runtime/ffmpeg/ffmpeg.exe'
if (-not (Test-Path -LiteralPath $Ffmpeg)) {
    if ($Offline) { throw "Offline setup: FFmpeg missing at $Ffmpeg" }
    $Zip = Join-Path $Root 'runtime/ffmpeg/ffmpeg.zip'
    $FfmpegUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'
    Invoke-WebRequest -Uri $FfmpegUrl -OutFile $Zip
    $Extract = Join-Path $Root 'runtime/ffmpeg/extracted'
    Expand-Archive -LiteralPath $Zip -DestinationPath $Extract -Force
    $Bin = Get-ChildItem -LiteralPath $Extract -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    if (-not $Bin) { throw 'Downloaded FFmpeg archive did not contain ffmpeg.exe.' }
    Copy-Item -LiteralPath (Join-Path $Bin.DirectoryName 'ffmpeg.exe') -Destination $Ffmpeg -Force
    Copy-Item -LiteralPath (Join-Path $Bin.DirectoryName 'ffprobe.exe') -Destination (Join-Path $Root 'runtime/ffmpeg/ffprobe.exe') -Force
    # The application only invokes ffmpeg and ffprobe. Do not retain the player,
    # downloaded archive, or duplicate extraction tree after a successful install.
    Remove-Item -LiteralPath $Zip -Force
    Remove-Item -LiteralPath $Extract -Recurse -Force
}
& $Ffmpeg -version | Select-Object -First 1
if ($LASTEXITCODE -ne 0) { throw 'Local FFmpeg validation failed.' }

$DeepFilter = Join-Path $Root 'runtime/deepfilternet/deep-filter.exe'
if (-not (Test-Path -LiteralPath $DeepFilter)) {
    if ($Offline) { throw "Offline setup: DeepFilterNet missing at $DeepFilter" }
    $DfRelease = Invoke-RestMethod -Uri 'https://api.github.com/repos/Rikorose/DeepFilterNet/releases/latest' -Headers @{ 'User-Agent'='whisper-persian-local' }
    $DfAsset = $DfRelease.assets | Where-Object { $_.name -match 'x86_64-pc-windows-msvc\.exe$' } | Select-Object -First 1
    if (-not $DfAsset) { throw 'Could not locate official DeepFilterNet Windows x64 release binary.' }
    Invoke-WebRequest -Uri $DfAsset.browser_download_url -OutFile $DeepFilter
    $DfRelease | Select-Object tag_name,published_at,html_url | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $Root 'runtime/deepfilternet/release.json')
}
& $DeepFilter --version
if ($LASTEXITCODE -ne 0) { throw 'DeepFilterNet validation failed.' }

& $VenvPython (Join-Path $Root 'src/download_models.py') --root $Root @($Offline ? '--local-only' : $null)
if ($LASTEXITCODE -ne 0) { throw 'CTranslate2 model download/validation failed.' }

& $VenvPython (Join-Path $Root 'src/download_semantic_encoder.py') --root $Root @($Offline ? '--local-only' : $null)
if ($LASTEXITCODE -ne 0) { throw 'Local semantic ONNX encoder download/validation failed.' }

$DictionaryArgs = @()
if ($Offline) { $DictionaryArgs += '-Offline' }
& (Join-Path $Root 'rebuild_drug_dictionary.ps1') @DictionaryArgs
if ($LASTEXITCODE -ne 0) { throw 'Combined licensed Persian drug dictionary build failed.' }

$CorpusArgs = @('--output-dir', (Join-Path $Root 'offline-corpus'))
if ($Offline) { $CorpusArgs += '--local-only' }
& $VenvPython (Join-Path $Root 'src/build_domain_corpus.py') @CorpusArgs
if ($LASTEXITCODE -ne 0) { throw 'Persian medical/daily n-gram corpus validation failed.' }

$PackageVersions = & $VenvPython -c "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ['faster-whisper','ctranslate2','huggingface-hub','numpy','onnxruntime','tokenizers']}))"
$Manifest = [ordered]@{
    created=(Get-Date).ToString('o'); cpu_threads=$Threads; nvidia=[bool]$Nvidia;
    device=$Device; compute_type=$ComputeType; packages=($PackageVersions | ConvertFrom-Json);
    deepfilternet='0.5.6';
    qwen=[ordered]@{
        repository='unsloth/Qwen3.5-35B-A3B-GGUF'; revision=$QwenRevision;
        file=$QwenFileName; sha256=$QwenSha256; bytes=(Get-Item -LiteralPath $QwenModel).Length;
        quantization='UD-Q4_K_L'; local_only_at_runtime=$true
    };
    llama_cpp=[ordered]@{ release=$LlamaRelease; backend=$LlamaRuntimeName; server=$LlamaServer }
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $Root 'runtime/setup-manifest.json')
Write-Host 'Setup and validation completed.'
