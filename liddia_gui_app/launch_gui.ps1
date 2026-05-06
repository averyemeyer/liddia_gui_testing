param(
    [string]$Port = "7961",
    [string]$Python = "python"
)

# Launch the modular LIDDIA GUI from this repository checkout.
# Example:
#   powershell -ExecutionPolicy Bypass -File .\liddia_gui_app\launch_gui.ps1

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $ScriptDir

Set-Location $ScriptDir
$env:GRADIO_SERVER_PORT = $Port

Write-Host "Repository: $RepoDir"
Write-Host "Python: $Python"
Write-Host "URL: http://127.0.0.1:$Port/"

& $Python -m liddia_gui.app
