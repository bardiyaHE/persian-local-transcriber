[CmdletBinding()]
param([int]$Port = 7860)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$App = Join-Path $Root 'src\web_app.py'
$Log = Join-Path $Root 'outputs\web-ui.log'
$Url = "http://127.0.0.1:$Port"
if (-not (Test-Path -LiteralPath $Python)) { throw 'محیط محلی پیدا نشد؛ ابتدا setup.ps1 را اجرا کنید.' }
$ProfileManifest = Join-Path $Root 'runtime\install-profile.json'
if (-not (Test-Path -LiteralPath $ProfileManifest)) {
    throw 'پروفایل نصب پیدا نشد؛ ابتدا setup.ps1 را اجرا کنید.'
}
$Profile = [string]((Get-Content -LiteralPath $ProfileManifest -Raw | ConvertFrom-Json).profile)
if ($Profile -eq 'full') {
    & (Join-Path $Root 'start_local_qwen.ps1')
}
$Running = $false
try { $Running = (Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch { $Running = $false }
if (-not $Running) {
    Start-Process -FilePath $Python -ArgumentList @($App, '--port', $Port) -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError (Join-Path $Root 'outputs\web-ui-error.log')
    for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
        Start-Sleep -Seconds 1
        try {
            if ((Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $Running = $true; break }
        } catch { }
    }
}
if (-not $Running) { throw "رابط اجرا نشد؛ گزارش را ببینید: $Log" }
Start-Process $Url
Write-Host "رابط آماده است: $Url"
