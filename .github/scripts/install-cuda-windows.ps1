# Installs a CUDA toolkit on Windows with NVIDIA's network installer, pinned by
# URL and SHA-256, with only the subpackages that ArrayFire's CUDA backend and its
# tests need to build. The names are NVIDIA's silent-install subpackage names
# ("Possible Subpackage Names" in the CUDA Installation Guide for Microsoft
# Windows). From CUDA 13.0 on, nvcc no longer includes crt (the headers such as
# crt/host_config.h) and nvvm (cicc): NVIDIA's redistrib_<version>.json lists
# cuda_crt and libnvvm apart from cuda_nvcc, and the installers carry crt_<version>
# and nvvm_<version> packages, though the guides list them only from 13.2 on.
#
# The toolkit goes to its default place, C:\Program Files\NVIDIA GPU Computing
# Toolkit\CUDA\v<version>. Under GitHub Actions, CUDA_PATH and the toolkit's bin
# directory are exported for the later steps.
#
# To add a version, take the newest patch release's network installer for x86_64
# from https://developer.download.nvidia.com/compute/cuda/<release>/network_installers/,
# check it against NVIDIA's docs/sidebar/md5sum.txt of that release, and pin its
# SHA-256.
#
# Usage: install-cuda-windows.ps1 -Version <major.minor>
param(
    [Parameter(Mandatory = $true)][string]$Version
)
$ErrorActionPreference = 'Stop'

$installers = @{
    '12.9' = @{ Release = '12.9.2'; File = 'cuda_12.9.2_windows_x86_64_network.exe'
                Sha256 = 'dd7b10659dd38c4551e8e0a5eaeb607e9b642dd01f803c45f561d4d18b04d88c' }
    '13.1' = @{ Release = '13.1.2'; File = 'cuda_13.1.2_windows_network.exe'
                Sha256 = '45d507ffb09f7821196f248c5473fbcc1ff62334b791292d9ec95767ec0f6564' }
    '13.4' = @{ Release = '13.4.3'; File = 'cuda_13.4.3_windows_x86_64_network.exe'
                Sha256 = 'c4f4dcde14545e878cdfa5769bac46d751d64ad9f1476cb83b59bbbec12ebf29' }
}
$subpackages = @('nvcc', 'cudart', 'thrust', 'cublas', 'cublas_dev', 'cufft', 'cufft_dev',
                 'cusolver', 'cusolver_dev', 'cusparse', 'cusparse_dev', 'nvrtc', 'nvrtc_dev',
                 'nvjitlink')
if ([version]$Version -ge [version]'13.0') { $subpackages += @('crt', 'nvvm') }

$pin = $installers[$Version]
if (-not $pin) {
    throw "No pinned CUDA installer for $Version (pinned: $($installers.Keys -join ', ')); add one to $PSCommandPath"
}
$url = "https://developer.download.nvidia.com/compute/cuda/$($pin.Release)/network_installers/$($pin.File)"
$installer = Join-Path ([System.IO.Path]::GetTempPath()) $pin.File
Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
$sha256 = (Get-FileHash -Algorithm SHA256 $installer).Hash
if ($sha256 -ne $pin.Sha256) {
    throw "$($pin.File) has SHA-256 $sha256, not the pinned $($pin.Sha256)"
}

$arguments = @('-s') + ($subpackages | ForEach-Object { "${_}_$Version" })
Write-Host "Installing CUDA $($pin.Release): $($arguments -join ' ')"
$process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru -NoNewWindow
if ($process.ExitCode) { throw "The CUDA installer failed with exit code $($process.ExitCode)" }

$root = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v$Version"
$nvcc = Join-Path $root 'bin\nvcc.exe'
if (-not (Test-Path $nvcc)) { throw "The installer reported success, but $nvcc is missing" }
& $nvcc --version
if ($LASTEXITCODE) { exit $LASTEXITCODE }

if ($env:GITHUB_ENV) {
    "CUDA_PATH=$root" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    (Join-Path $root 'bin') | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
}
