# Sets up the environment of Visual Studio's x64 developer prompt (vcvars64.bat):
# the variables that the Ninja generator, cl and nvcc need to find MSVC, which
# the Visual Studio generator finds on its own. Under GitHub Actions they are
# exported for the later steps; elsewhere they are printed.
#
# Usage: msvc-env-windows.ps1
$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'No Visual Studio with the x64 C++ tools found' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
Write-Host "Using $vcvars"

$changed = @()
foreach ($line in (cmd /c "`"$vcvars`" > nul && set")) {
    if ($line -match '^([^=]+)=(.*)$' -and [Environment]::GetEnvironmentVariable($Matches[1]) -ne $Matches[2]) {
        $changed += "$($Matches[1])=$($Matches[2])"
    }
}
if (-not $changed) { throw "$vcvars changed no variable" }
if ($env:GITHUB_ENV) {
    $changed | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    Write-Host "Exported $($changed.Count) variables: $(($changed | ForEach-Object { $_.Split('=')[0] }) -join ', ')"
} else {
    $changed
}
