#!/usr/bin/env bash
# Installs the packages for an ArrayFire build, the docs, the lint check or the
# workflow checks on Ubuntu 22.04, 24.04 or 26.04. Runs as root or through sudo.
# With --runtime, --cuda and --opencl install only what built CUDA or OpenCL tests
# need to run: the libraries, ctest and zstd, without compilers or headers.
#
# Usage: install-deps-ubuntu.sh [--compiler gcc|clang|icx] [--blas openblas|mkl]
#                               [--oneapi <version>] [--cuda <version>] [--opencl]
#                               [--runtime] [--cmake <version>] [--docs] [--lint]
#                               [--workflows]
set -euo pipefail

oneapi_version=2026.2
doxygen_version=1.9.7
doxygen_sha256=8bcb7cbdb9d8ae78d0bf65a383bc533db5232e2527ab10743f1e58cba92a7d43
actionlint_version=1.7.9
actionlint_sha256=233b280d05e100837f4af1433c7b40a5dcb306e3aa68fb4f17f8a7f45a7df7b4
shellcheck_version=0.11.0
shellcheck_sha256=8c3be12b05d5c177a04c29e3c78ce89ac86f1595681cab149b65b97c4e227198
declare -A cmake_sha256=(
  [3.16.3]=3e15dadfec8d54eda39c2f266fc1e571c1b88bf32f9d221c8a039b07234206fa
)

compiler=
blas=
cuda_version=
opencl=
runtime=
cmake_version=
docs=
lint=
workflows=
while [[ $# -gt 0 ]]; do
  case $1 in
    --compiler) compiler=$2; shift 2 ;;
    --blas) blas=$2; shift 2 ;;
    --oneapi) oneapi_version=$2; shift 2 ;;
    --cuda) cuda_version=$2; shift 2 ;;
    --opencl) opencl=1; shift ;;
    --runtime) runtime=1; shift ;;
    --cmake) cmake_version=$2; shift 2 ;;
    --docs) docs=1; shift ;;
    --lint) lint=1; shift ;;
    --workflows) workflows=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

sudo=
[[ $(id -u) -eq 0 ]] || sudo=sudo
export DEBIAN_FRONTEND=noninteractive

packages=(ca-certificates curl git)
[[ -n $lint ]] && packages+=(python3 python3-venv)
[[ -n $workflows ]] && packages+=(python3 xz-utils)
# ci.sh restores the tests' Matrix Market data with check-matrix-data.py before a build.
[[ -n $compiler ]] && packages+=(python3)
if [[ -n $compiler || -n $docs ]]; then
  packages+=(build-essential cmake ninja-build libboost-dev)
fi
if [[ -n $compiler ]]; then
  packages+=(ccache pkg-config libfftw3-dev liblapacke-dev libfreeimage-dev
             libglfw3-dev libfreetype-dev libfontconfig1-dev)
fi
case $compiler in
  gcc|icx|'') ;;
  clang) packages+=(clang) ;;
  *) echo "unknown compiler: $compiler" >&2; exit 2 ;;
esac
case $blas in
  mkl|'') ;;
  openblas) packages+=(libopenblas-dev) ;;
  *) echo "unknown BLAS: $blas" >&2; exit 2 ;;
esac
if [[ -n $runtime ]]; then
  # ctest comes with cmake. ArrayFire loads FreeImage at run time as libfreeimage.so,
  # a name only the -dev package has. The OpenCL tests need the ICD loader, BLAS and
  # LAPACK, and clinfo to list the platforms.
  packages+=(cmake zstd libfreeimage-dev)
  [[ -n $opencl ]] && packages+=(ocl-icd-libopencl1 clinfo libopenblas0 liblapacke)
else
  [[ -n $opencl ]] && packages+=(ocl-icd-opencl-dev opencl-headers)
  # The GPU build jobs archive their tests with zstd.
  [[ -n $cuda_version || -n $opencl ]] && packages+=(zstd)
fi
[[ $compiler == icx || $blas == mkl || -n $cuda_version ]] && packages+=(gnupg)

$sudo apt-get update -q
$sudo apt-get install -y -q --no-install-recommends "${packages[@]}"

if [[ $compiler == icx || $blas == mkl ]]; then
  keyring=/usr/share/keyrings/oneapi-archive-keyring.gpg
  curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | $sudo tee $keyring > /dev/null
  echo "deb [signed-by=$keyring] https://apt.repos.intel.com/oneapi all main" \
    | $sudo tee /etc/apt/sources.list.d/oneAPI.list > /dev/null
  oneapi_packages=("intel-oneapi-mkl-devel-$oneapi_version")
  [[ $compiler == icx ]] && oneapi_packages+=("intel-oneapi-compiler-dpcpp-cpp-$oneapi_version")
  $sudo apt-get update -q
  $sudo apt-get install -y -q "${oneapi_packages[@]}"
fi

if [[ -n $cuda_version ]]; then
  # The toolkit pieces ArrayFire's CUDA backend builds against; no driver is needed.
  # CCCL comes with cudart-dev, as cuda-cccl or (from 13.3) cccl.
  # shellcheck disable=SC1091
  repo=ubuntu$(. /etc/os-release && echo "${VERSION_ID//./}")
  keyring=/usr/share/keyrings/cuda-archive-keyring.gpg
  curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/3bf863cc.pub" \
    | gpg --dearmor | $sudo tee $keyring > /dev/null
  echo "deb [signed-by=$keyring] https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/ /" \
    | $sudo tee /etc/apt/sources.list.d/cuda.list > /dev/null
  v=${cuda_version//./-}
  $sudo apt-get update -q
  if [[ -n $runtime ]]; then
    # The libraries ArrayFire's CUDA backend loads; it embeds the headers its
    # run-time compilation (NVRTC) needs.
    $sudo apt-get install -y -q --no-install-recommends \
      "cuda-nvrtc-$v" "libcublas-$v" "libcufft-$v" "libcusolver-$v" "libcusparse-$v" \
      "libnvjitlink-$v"
    $sudo ln -sfn "/usr/local/cuda-$cuda_version" /usr/local/cuda
    # ArrayFire loads cuSPARSE at run time as libcusparse.so, a name only the 200 MB
    # libcusparse-dev package has; link it to the library's soname (libcusparse.so.12
    # sorts before libcusparse.so.12.5.10.65) instead.
    sonames=(/usr/local/cuda/lib64/libcusparse.so.[0-9]*)
    $sudo ln -sfn "${sonames[0]##*/}" /usr/local/cuda/lib64/libcusparse.so
    $sudo ldconfig
  else
    $sudo apt-get install -y -q --no-install-recommends \
      "cuda-nvcc-$v" "cuda-cudart-dev-$v" "cuda-nvrtc-dev-$v" \
      "cuda-driver-dev-$v" "cuda-profiler-api-$v" "cuda-nvtx-$v" "cuda-nvprune-$v" \
      "libcublas-dev-$v" "libcufft-dev-$v" "libcusolver-dev-$v" "libcusparse-dev-$v" \
      "libcurand-dev-$v" "libnvjitlink-dev-$v"
    $sudo ln -sfn "/usr/local/cuda-$cuda_version" /usr/local/cuda
    /usr/local/cuda/bin/nvcc --version | tail -n 2
  fi
fi

# Downloads a release archive, checks its SHA-256 and unpacks it into /opt/<name>.
install_archive() {
  local url=$1 sha256=$2 name=$3 tmp
  tmp=$(mktemp)
  curl -fsSL -o "$tmp" "$url"
  echo "$sha256  $tmp" | sha256sum -c --quiet
  $sudo mkdir -p "/opt/$name"
  $sudo tar -xzf "$tmp" -C "/opt/$name" --strip-components 1
  rm -f "$tmp"
}

if [[ -n $cmake_version ]]; then
  sha=${cmake_sha256[$cmake_version]:-}
  [[ -n $sha ]] || { echo "no checksum for CMake $cmake_version" >&2; exit 2; }
  install_archive \
    "https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-Linux-x86_64.tar.gz" \
    "$sha" "cmake-$cmake_version"
  "/opt/cmake-$cmake_version/bin/cmake" --version | head -n 1
fi

if [[ -n $docs ]]; then
  install_archive \
    "https://github.com/doxygen/doxygen/releases/download/Release_${doxygen_version//./_}/doxygen-$doxygen_version.linux.bin.tar.gz" \
    "$doxygen_sha256" "doxygen-$doxygen_version"
  $sudo ln -sf "/opt/doxygen-$doxygen_version/bin/doxygen" /usr/local/bin/doxygen
  echo "doxygen $(doxygen --version)"
fi

# Downloads a release archive, checks its SHA-256 and installs one program from it.
install_program() {
  local url=$1 sha256=$2 member=$3 tmp
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/archive" "$url"
  echo "$sha256  $tmp/archive" | sha256sum -c --quiet
  tar -xf "$tmp/archive" -C "$tmp" "$member"
  $sudo install -m 755 "$tmp/$member" "/usr/local/bin/${member##*/}"
  rm -rf "$tmp"
}

if [[ -n $workflows ]]; then
  install_program \
    "https://github.com/rhysd/actionlint/releases/download/v$actionlint_version/actionlint_${actionlint_version}_linux_amd64.tar.gz" \
    "$actionlint_sha256" actionlint
  install_program \
    "https://github.com/koalaman/shellcheck/releases/download/v$shellcheck_version/shellcheck-v$shellcheck_version.linux.x86_64.tar.xz" \
    "$shellcheck_sha256" "shellcheck-v$shellcheck_version/shellcheck"
  echo "actionlint $(actionlint --version | head -n 1), shellcheck $(shellcheck --version | sed -n 2p)"
fi
