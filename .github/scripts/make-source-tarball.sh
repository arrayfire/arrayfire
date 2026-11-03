#!/usr/bin/env bash
# Creates arrayfire-full-<version>.tar.bz2 next to the source directory: the
# sources plus every dependency that CMake would otherwise download, so the
# tarball builds without network access.
#
# Usage: make-source-tarball.sh <source directory> <version>
#   The source directory is renamed to arrayfire-full-<version> if needed. It is
#   modified in place, so pass a throwaway checkout.
set -euo pipefail

source_dir=$1
version=$2
forge_version=1.0.8
name=arrayfire-full-$version

sudo=
[[ $(id -u) -eq 0 ]] || sudo=sudo
export DEBIAN_FRONTEND=noninteractive

# Configuring with the CUDA and OpenCL toolkits present makes CMake fetch the
# sources of every backend's dependencies.
$sudo apt-get update -q
$sudo apt-get install -y -q --no-install-recommends \
  build-essential bzip2 ca-certificates cmake curl git libboost-dev \
  libfftw3-dev libfontconfig1-dev libglfw3-dev liblapacke-dev libopenblas-dev \
  nvidia-cuda-toolkit ocl-icd-opencl-dev

parent=$(cd "$(dirname "$source_dir")" && pwd)
if [[ $(basename "$source_dir") != "$name" ]]; then
  mv "$source_dir" "$parent/$name"
fi
cd "$parent/$name"

cmake -S . -B build -DAF_BUILD_FORGE=ON -DAF_COMPUTE_LIBRARY=FFTW/LAPACK/BLAS \
  -DAF_TEST_WITH_MTX_FILES=OFF

rm -rf .git .github .gitmodules
cd build/extern
rm -rf ./*-build ./*-subbuild matrixmarket
for dep in */; do
  rm -rf "$dep.git" "$dep.gitattributes" "$dep.gitmodules"
done
cp -r ./* ../../extern/
cd ../..

curl -fsSL "https://github.com/arrayfire/forge/releases/download/v$forge_version/forge-full-$forge_version.tar.bz2" \
  | tar -xj
mv "forge-full-$forge_version" extern/af_forge-src
rm -rf build

cd "$parent"
tar -cjf "$name.tar.bz2" "$name"
echo "Created $parent/$name.tar.bz2"
