#!/usr/bin/env bash
# Configures the CUDA backend the way users and the release installers get it,
# with ArrayFire's default architecture list and AF_INSTALL_STANDALONE=ON, and
# checks two things that the CI builds' explicit architecture list and plain
# install would hide:
#
# - The default architecture list: prints the flags CMake chose and compiles one
#   CUDA object of afcuda with them, which fails when the list holds an
#   architecture the toolkit dropped. Without a GPU, the default ("Auto") is the
#   common list for the toolkit.
# - The bundled CUDA libraries: installs the cuda_dependencies component into a
#   temporary prefix and checks that every library afcu_collect_libs bundles on
#   Linux (cuFFT, cuBLAS, cuBLASLt, cuSOLVER, cuSPARSE, nvJitLink, NVRTC and its
#   builtins) is there once, under the toolkit's soname, which is the name
#   libafcuda loads it by.
#
# Only a configure and one compile: it needs the CUDA toolkit, not a build or a GPU.
#
# Usage: check-cuda-release.sh <source dir> <build dir>
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 <source dir> <build dir>" >&2; exit 2; }
source=$1 build=$2
cuda=/usr/local/cuda
failed=

cmake -S "$source" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
  -DAF_BUILD_CPU=OFF -DAF_BUILD_CUDA=ON -DAF_BUILD_OPENCL=OFF -DAF_BUILD_ONEAPI=OFF \
  -DAF_BUILD_UNIFIED=OFF -DAF_BUILD_FORGE=OFF -DAF_BUILD_EXAMPLES=OFF -DAF_BUILD_DOCS=OFF \
  -DBUILD_TESTING=OFF -DAF_WITH_CUDNN=OFF -DAF_INSTALL_STANDALONE=ON > "$build.configure.log" 2>&1 ||
  { tail -n 30 "$build.configure.log"; echo "The configure failed; see $build.configure.log" >&2; exit 1; }
grep -m 1 -E 'GPU detection|common architectures' "$build.configure.log" || true

echo "=== The default architecture list"
object=src/backend/cuda/CMakeFiles/afcuda.dir/cufft.cu.o
flags=$(grep -A 8 "^build $object:" "$build/build.ninja" | grep -m 1 '^ *FLAGS = ' |
        grep -o -- '--generate-code=arch=[^ "]*\|-gencode[= ]arch=[^ "]*' | sed 's/^--generate-code=//' || true)
echo "CUDA_architecture_build_targets=$(sed -n 's/^CUDA_architecture_build_targets:[A-Z]*=//p' "$build/CMakeCache.txt")"
echo "Flags for $object:"
while IFS= read -r flag; do echo "  $flag"; done <<< "${flags:-(none found)}"
if ninja -C "$build" "$object" > "$build.compile.log" 2>&1; then
  echo "Compiling $object with them: passed"
else
  grep -E 'nvcc fatal|error' "$build.compile.log" | head -n 5
  echo "Compiling $object with them: FAILED" >&2
  failed=1
fi

echo "=== The bundled CUDA libraries (AF_INSTALL_STANDALONE=ON, component cuda_dependencies)"
prefix=$(mktemp -d)
if ! cmake --install "$build" --component cuda_dependencies --prefix "$prefix" > "$build.install.log" 2>&1; then
  grep -E 'CMake Error|cannot find|not found' "$build.install.log" | head -n 5
  echo "Installing cuda_dependencies: FAILED" >&2
  failed=1
fi
soname() { { readelf -d "$1" 2> /dev/null || true; } | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p'; }
printf '%-15s %-28s %-28s %s\n' library "toolkit soname" bundled "bundled file's soname"
for lib in cufft cublas cublasLt cusolver cusparse nvJitLink nvrtc nvrtc-builtins; do
  expected=$(soname "$(readlink -f "$cuda/lib64/lib$lib.so")")
  expected=${expected:-"(no lib$lib.so)"}
  bundled=$(cd "$prefix" && find . -name "lib$lib.so*" | sed 's|^\./||' | sort | tr '\n' ' ')
  bundled=${bundled% }
  if [[ -z $bundled ]]; then
    printf '%-15s %-28s %-28s %s\n' "$lib" "$expected" "MISSING" "-"
    failed=1
  elif [[ $bundled == *' '* ]]; then
    printf '%-15s %-28s %-28s %s\n' "$lib" "$expected" "$bundled" "(more than one)"
    failed=1
  else
    actual=$(soname "$prefix/$bundled")
    status=
    [[ ${bundled##*/} == "$expected" && $actual == "$expected" ]] || { status="  <- WRONG NAME"; failed=1; }
    printf '%-15s %-28s %-28s %s%s\n' "$lib" "$expected" "$bundled" "$actual" "$status"
  fi
done
others=$(cd "$prefix" && find . -type f | sed 's|^\./||' |
         grep -v -E '/lib(cufft|cublas|cublasLt|cusolver|cusparse|nvJitLink|nvrtc|nvrtc-builtins)\.so' || true)
[[ -z $others ]] || echo "Also bundled: $others"
rm -rf "$prefix"

if [[ -n $failed ]]; then
  echo "The release configuration check failed." >&2
  exit 1
fi
echo "The release configuration check passed."
