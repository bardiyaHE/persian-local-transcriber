[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$AudioFile,
    [string]$RunId = ''
)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$ResolvedAudio = (Resolve-Path -LiteralPath $AudioFile -ErrorAction Stop).Path
$Python = Join-Path $Root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Local environment is missing. Run setup.ps1 first.' }
$Threads = [Environment]::ProcessorCount
if (-not $RunId) { $RunId = 'cli-' + (Get-Date -Format 'yyyyMMdd-HHmmss-ffffff') }
if ($RunId -notmatch '^[A-Za-z0-9._-]+$') { throw 'RunId may contain only letters, digits, dot, underscore and hyphen.' }
$PipelineStarted = Get-Date
$Nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($Nvidia) {
    $Device = 'cuda'; $ComputeType = 'float16'
    $CudaRoot = Join-Path $Root 'runtime/cuda-libs'
    $CudaDllDirs = Get-ChildItem -LiteralPath $CudaRoot -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq 'bin' }
    foreach ($Dir in $CudaDllDirs) { $env:PATH = $Dir.FullName + ';' + $env:PATH }
    if (-not $CudaDllDirs) { throw 'NVIDIA detected but local CUDA 12/cuDNN 9 DLL directories are missing; refusing CPU fallback.' }
} else {
    $Device = 'cpu'; $ComputeType = 'int8'
}
& $Python (Join-Path $Root 'src/pipeline.py') --audio $ResolvedAudio --root $Root --device $Device --compute-type $ComputeType --threads $Threads --run-id $RunId --adaptive-turbo
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }
$RunDir = Join-Path $Root (Join-Path 'outputs' $RunId)
$MedicalIndex = Join-Path $Root 'offline-lexicon/combined-medical-drug-index.json'
$CorpusIndex = Join-Path $Root 'offline-corpus/domain-ngrams-v1.sqlite3'
$EncoderDir = Join-Path $Root 'models/semantic-encoder-v1'
& $Python (Join-Path $Root 'src/consensus_v9_medical_drugs.py') --run-dir $RunDir --medical-index $MedicalIndex --corpus-index $CorpusIndex --encoder-dir $EncoderDir
if ($LASTEXITCODE -ne 0) { throw "Consensus failed with exit code $LASTEXITCODE" }
& (Join-Path $Root 'run_local_qwen_v10.ps1') -RunDir $RunDir
& (Join-Path $Root 'run_local_qwen_summary_v11.ps1') -RunDir $RunDir
$ElapsedSeconds = ((Get-Date) - $PipelineStarted).TotalSeconds
$TurboRaw = Join-Path $RunDir 'hypotheses/large-v3-turbo__raw'
$TurboJson = Get-ChildItem -LiteralPath $TurboRaw -Filter '*.json' -File | Select-Object -First 1
$AudioSeconds = $null
if ($TurboJson) {
    $AudioSeconds = [double]((Get-Content -LiteralPath $TurboJson.FullName -Raw | ConvertFrom-Json).duration)
}
$QwenSummary = Get-Content -LiteralPath (Join-Path $RunDir 'final-delivery/10-local-qwen-reranker/summary-v10.json') -Raw | ConvertFrom-Json
$MedicalSummary = Get-Content -LiteralPath (Join-Path $RunDir 'final-delivery/11-local-qwen-summary/summary-v11.json') -Raw | ConvertFrom-Json
$Gpu = $null
if ($Nvidia) {
    $Gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | Select-Object -First 1).Trim()
}
$Benchmark = [ordered]@{
    run_id = $RunId
    recorded_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    audio_duration_seconds = $AudioSeconds
    end_to_end_seconds = [Math]::Round($ElapsedSeconds, 3)
    under_60_seconds = ($ElapsedSeconds -lt 60.0)
    whisper_device = $Device
    whisper_compute_type = $ComputeType
    gpu = $Gpu
    qwen_runtime_seconds = $QwenSummary.runtime_seconds
    qwen_model_latency_seconds = $QwenSummary.model.latency_seconds
    qwen_call_count = $QwenSummary.model.call_count
    summary_runtime_seconds = $MedicalSummary.runtime_seconds
    summary_model_latency_seconds = $MedicalSummary.model.latency_seconds
    summary_accepted = $MedicalSummary.accepted
    external_api_used = $QwenSummary.external_api_used_at_runtime
    free_text_generation_enters_output = $QwenSummary.free_text_generation_enters_output
    generated_summary_enters_transcript = $MedicalSummary.generated_summary_enters_transcript
}
$Benchmark | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RunDir 'runtime-benchmark.json') -Encoding utf8
Write-Host ("End-to-end seconds: {0:N2}" -f $ElapsedSeconds)
Write-Host "Benchmark: $(Join-Path $RunDir 'runtime-benchmark.json')"
Write-Host "Local Qwen V10 result: $(Join-Path $RunDir 'final-delivery/10-local-qwen-reranker/final-v10.txt')"
Write-Host "Local Qwen V11 summary: $(Join-Path $RunDir 'final-delivery/11-local-qwen-summary/final-summary-v11.txt')"
Write-Host "Review report: $(Join-Path $RunDir 'final-delivery/10-local-qwen-reranker/review-v10.md')"
