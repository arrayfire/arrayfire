# Clones vcpkg at a pinned release and bootstraps it. CMake installs the
# dependencies listed in vcpkg.json (manifest mode) during configure, so set
# VCPKG_ROOT to the same directory before configuring.
#
# The port versions come from vcpkg.json's builtin-baseline, whichever vcpkg commit
# is checked out. This commit fixes only the vcpkg tool and its helper scripts,
# which download build tools such as MSYS2 packages by exact version; MSYS2 drops
# old versions, so the helpers have to stay recent while the baseline stays put.
# To update, take the latest release from https://github.com/microsoft/vcpkg/releases
# and set $vcpkgCommit to its commit (git rev-parse <tag>^{commit}).
#
# Usage: install-deps-windows.ps1 [-VcpkgRoot <dir>]   (default: $env:VCPKG_ROOT or ~/vcpkg)
param(
    [string]$VcpkgRoot = $(if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { Join-Path $HOME 'vcpkg' })
)
$ErrorActionPreference = 'Stop'

# vcpkg 2026.10.16
$vcpkgCommit = '9eec002df1cd109f8f104cd966a763daf9751489'

if (-not (Test-Path (Join-Path $VcpkgRoot '.git'))) {
    git clone https://github.com/microsoft/vcpkg.git $VcpkgRoot
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
git -C $VcpkgRoot fetch --quiet origin $vcpkgCommit
if ($LASTEXITCODE) { exit $LASTEXITCODE }
git -C $VcpkgRoot checkout --quiet --detach $vcpkgCommit
if ($LASTEXITCODE) { exit $LASTEXITCODE }

& (Join-Path $VcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics
if ($LASTEXITCODE) { exit $LASTEXITCODE }

Write-Host "vcpkg $vcpkgCommit is ready in $VcpkgRoot; set VCPKG_ROOT to it before configuring."
