[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$PidFile = Join-Path $Root 'runtime/local-qwen-server.pid'
if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host 'No local Qwen PID file exists.'
    exit 0
}
$ServerPid = [int](Get-Content -Raw -LiteralPath $PidFile).Trim()
$Process = Get-Process -Id $ServerPid -ErrorAction SilentlyContinue
if ($Process) {
    $Executable = $Process.Path
    $ExpectedRoot = [System.IO.Path]::GetFullPath((Join-Path $Root 'runtime/llama.cpp'))
    if (-not $Executable -or -not [System.IO.Path]::GetFullPath($Executable).StartsWith($ExpectedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PID $ServerPid does not belong to this project's llama.cpp runtime; refusing to stop it."
    }
    Stop-Process -Id $ServerPid -Force
    Write-Host "Stopped local Qwen process $ServerPid."
}
Remove-Item -LiteralPath $PidFile -Force
