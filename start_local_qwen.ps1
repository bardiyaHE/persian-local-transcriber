[CmdletBinding()]
param(
    [int]$Port = 18080,
    [int]$ContextSize = 4096,
    [int]$Threads = 0
)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Model = Join-Path $Root 'models/qwen3.5-35b-a3b-gguf/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf'
$ModelAlias = 'local-qwen3.5-35b-a3b'
$PidFile = Join-Path $Root 'runtime/local-qwen-server.pid'
$StdoutLog = Join-Path $Root 'runtime/local-qwen-server.stdout.log'
$StderrLog = Join-Path $Root 'runtime/local-qwen-server.stderr.log'
$HealthUrl = "http://127.0.0.1:$Port/health"

try {
    $Health = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
    if ($Health.StatusCode -eq 200) {
        $Models = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2
        $LoadedIds = @($Models.data | ForEach-Object { [string]$_.id })
        if ($LoadedIds -contains $ModelAlias) {
            Write-Host "Local Qwen 35B-A3B is already ready at $HealthUrl"
            exit 0
        }
        throw "Port $Port is occupied by a different model: $($LoadedIds -join ', ')"
    }
} catch { }

if (-not (Test-Path -LiteralPath $Model)) {
    throw "Local Qwen model is missing: $Model. Run setup.ps1 first."
}
$Nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$RuntimeName = if ($Nvidia) { 'cuda' } else { 'cpu' }
$Server = Join-Path $Root "runtime/llama.cpp/$RuntimeName/llama-server.exe"
if (-not (Test-Path -LiteralPath $Server)) {
    throw "llama.cpp $RuntimeName runtime is missing: $Server. Run setup.ps1 first; NVIDIA never falls back silently to CPU."
}
if ($Threads -le 0) {
    $Threads = [Math]::Max(1, [Math]::Min(16, [Environment]::ProcessorCount))
}
$GpuLayers = if ($Nvidia) { 'all' } else { '0' }
$Arguments = @(
    '--model', $Model,
    '--alias', $ModelAlias,
    '--host', '127.0.0.1',
    '--port', [string]$Port,
    '--ctx-size', [string]$ContextSize,
    '--threads', [string]$Threads,
    '--threads-batch', [string]$Threads,
    '--n-gpu-layers', $GpuLayers,
    '--parallel', '1',
    '--jinja',
    '--reasoning', 'off',
    '--cors-origins', 'localhost',
    '--no-slots',
    '--no-webui'
)
$Process = Start-Process -FilePath $Server -ArgumentList $Arguments -WorkingDirectory $Root `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
$Process.Id | Set-Content -LiteralPath $PidFile -Encoding ascii

$Ready = $false
for ($Attempt = 0; $Attempt -lt 120; $Attempt++) {
    if ($Process.HasExited) {
        $Tail = if (Test-Path -LiteralPath $StderrLog) {
            (Get-Content -LiteralPath $StderrLog -Tail 80) -join "`n"
        } else { 'No stderr log was written.' }
        throw "Local Qwen exited during startup.`n$Tail"
    }
    Start-Sleep -Seconds 1
    try {
        if ((Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) {
            $Ready = $true
            break
        }
    } catch { }
}
if (-not $Ready) {
    Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    throw "Local Qwen did not become healthy within 120 seconds. See $StderrLog"
}
Write-Host "Local Qwen 35B-A3B ready: backend=$RuntimeName pid=$($Process.Id) url=$HealthUrl"
