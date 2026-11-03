#!/usr/bin/env bash
# Runs one CI job: installs its dependencies, then runs its CMake workflow
# preset or check. CI runs it inside the job's container image; --docker runs
# it the same way in a fresh local container.
#
# A job builds and tests the source tree in AF_CI_SOURCE_DIR (default: this
# repository) with this repository's scripts and CI settings, so it can test any
# commit. AF_CI_CUDA_VERSION (default 12.9) selects the CUDA toolkit of the
# linux-cuda* jobs, and AF_CI_CUDA_ARCHITECTURES (default 7.5;8.9) the GPU
# architectures of linux-cuda-gpu-build. windows-cuda needs the toolkit installed
# first (install-cuda-windows.ps1) and Visual Studio's x64 developer environment
# (msvc-env-windows.ps1).
# A *-gpu-build job builds the tests into an archive, and its
# *-gpu-test job runs them with the host's GPU (docker run --gpus all), with only
# the libraries they need. With AF_CI_CLEAN_EXPORT set, --docker builds a copy of the
# tree's tracked files without .git instead of mounting the tree itself.
# The Matrix Market tests run once .github/fixtures/matrixmarket.json pins an
# arrayfire-data commit (check-matrix-data.py). AF_CI_MTX_SOURCE turns them on with
# the matrices from another URL, such as file:///path/to/arrayfire-data/matrixmarket/
# (with --docker, a path inside the container), and AF_CI_MTX_CACHE keeps the
# downloads in a directory.
# It must run on macOS's bash 3.2: the macOS job runs it on the host.
#
# Usage: ci.sh [--docker] [--no-deps] <job>
#        ci.sh --list
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
scripts=$root/.github/scripts
floor_cmake=3.16.3
cuda_version=${AF_CI_CUDA_VERSION:-12.9}

# job, container image, arguments for the dependency script
jobs="
lint                          ubuntu:24.04  --lint
docs                          ubuntu:24.04  --docs
workflows                     ubuntu:24.04  --workflows
linux-gcc-openblas            ubuntu:24.04  --compiler gcc --blas openblas
linux-clang-openblas          ubuntu:24.04  --compiler clang --blas openblas
linux-icx-mkl                 ubuntu:24.04  --compiler icx --blas mkl
linux-gcc-openblas-floor      ubuntu:22.04  --compiler gcc --blas openblas --cmake $floor_cmake
linux-gcc-openblas-next       ubuntu:26.04  --compiler gcc --blas openblas
linux-cuda                    ubuntu:24.04  --compiler gcc --cuda $cuda_version
linux-opencl                  ubuntu:24.04  --compiler gcc --blas openblas --opencl
linux-oneapi                  ubuntu:24.04  --compiler icx --blas mkl --opencl --oneapi 2025.3
linux-cuda-gpu-build          ubuntu:24.04  --compiler gcc --cuda $cuda_version
linux-cuda-gpu-test           ubuntu:24.04  --runtime --cuda $cuda_version
linux-opencl-gpu-build        ubuntu:24.04  --compiler gcc --blas openblas --opencl
linux-opencl-gpu-test         ubuntu:24.04  --runtime --opencl
macos-arm64-clang-accelerate  -
windows-msvc-openblas         -
windows-cuda                  -
"

usage() {
  echo "usage: $0 [--docker] [--no-deps] <job>" >&2
  echo "       $0 --list" >&2
  exit 2
}

docker=
deps=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --list) echo "$jobs" | awk 'NF { print $1, ($2 == "-" ? "(host)" : $2) }'; exit 0 ;;
    --docker) docker=1; shift ;;
    --no-deps) deps=; shift ;;
    -*) usage ;;
    *) break ;;
  esac
done
[[ $# -eq 1 ]] || usage
job=$1
spec=$(echo "$jobs" | awk -v job="$job" '$1 == job')
[[ -n $spec ]] || { echo "unknown job: $job (see --list)" >&2; exit 2; }
read -r _ image deps_args <<< "$spec"

cd "$root"
source=$(cd "${AF_CI_SOURCE_DIR:-$root}" && pwd)
# A *-gpu-build job and its *-gpu-test job share build/ci-linux-<backend>-gpu.
gpu_build=$source/build/ci-${job%-*}
if [[ -n ${AF_CI_CLEAN_EXPORT:-} && -z $docker ]]; then
  echo "AF_CI_CLEAN_EXPORT needs --docker: a build on the host could reach the tree's credentials" >&2
  exit 2
fi

if [[ -n $docker ]]; then
  [[ $image != - ]] || { echo "$job runs on the host, not in a container" >&2; exit 2; }
  # ccache lives in a named volume, or in a host directory (AF_CI_CCACHE_DIR)
  # that CI can save between runs.
  ccache=${AF_CI_CCACHE_VOLUME:-arrayfire-ccache}
  if [[ -n ${AF_CI_CCACHE_DIR:-} ]]; then
    mkdir -p "$AF_CI_CCACHE_DIR"
    ccache=$(cd "$AF_CI_CCACHE_DIR" && pwd)
  fi
  # So are the downloaded Matrix Market data, when AF_CI_MTX_CACHE names a directory.
  mtx_cache=()
  if [[ -n ${AF_CI_MTX_CACHE:-} ]]; then
    mkdir -p "$AF_CI_MTX_CACHE"
    mtx_cache=(-v "$(cd "$AF_CI_MTX_CACHE" && pwd):/mtx-cache" -e AF_CI_MTX_CACHE=/mtx-cache)
  fi
  # Ninja's default (CPUs + 2) can run out of memory. Unless CMAKE_BUILD_PARALLEL_LEVEL
  # is set, run as many compiles as fit the container's --memory limit, or 3/4 of
  # Docker's memory without one: about 2.5 GiB each for SYCL and 0.75 GiB for the
  # rest. Never more than the CPUs, nor 8, the most every job was measured with.
  # Jobs that compile nothing skip this.
  if [[ -z ${CMAKE_BUILD_PARALLEL_LEVEL:-} && ! $job =~ ^(lint|docs|workflows|.*-gpu-test)$ ]]; then
    read -r cpus mib <<< "$(docker info --format '{{.NCPU}} {{.MemTotal}}')"
    mib=$((mib * 3 / 4 / 1048576))
    memory="3/4 of Docker's memory"
    args=$(printf ' %s ' "${AF_CI_DOCKER_ARGS:-}" | tr '[:upper:]' '[:lower:]')
    if [[ $args =~ [[:space:]](--memory|-m)[=[:space:]]*([0-9]+)([kmg]?)b?[[:space:]] ]]; then
      case ${BASH_REMATCH[3]} in
        g) mib=$((BASH_REMATCH[2] * 1024)) ;;
        m) mib=${BASH_REMATCH[2]} ;;
        k) mib=$((BASH_REMATCH[2] / 1024)) ;;
        *) mib=$((BASH_REMATCH[2] / 1048576)) ;;
      esac
      memory="the --memory limit"
    fi
    [[ $args =~ [[:space:]]--cpus[=[:space:]]*([0-9]+) ]] && cpus=${BASH_REMATCH[1]}
    per_compile=768 per_compile_gib=0.75
    [[ $job == linux-oneapi ]] && per_compile=2560 per_compile_gib=2.5
    parallel=$((mib / per_compile))
    ((parallel > cpus)) && parallel=$cpus
    ((parallel > 8)) && parallel=8
    ((parallel < 1)) && parallel=1
    export CMAKE_BUILD_PARALLEL_LEVEL=$parallel
    echo "Building with $parallel parallel jobs: $mib MiB ($memory) / $per_compile_gib GiB" \
      "per compile = $((mib / per_compile)), $cpus CPUs, at most 8." \
      "Set CMAKE_BUILD_PARALLEL_LEVEL to override."
  fi
  # A working tree that may hold credentials, such as the one an agent edits and
  # pushes from, is never mounted. The build gets a copy of its tracked files
  # without .git (no untracked files either), in a new directory under
  # AF_CI_EXPORT_DIR, so it can neither read the tree's git config nor change the
  # tree. The lint job gets the diff of the changed lines, computed here.
  lint_diff=()
  if [[ -n ${AF_CI_CLEAN_EXPORT:-} ]]; then
    export_dir=$(mktemp -d "${AF_CI_EXPORT_DIR:-${TMPDIR:-/tmp}}/arrayfire-ci.XXXXXX")
    mkdir "$export_dir/src"
    git --no-optional-locks -C "$source" ls-files -z --cached |
      (cd "$source" && while IFS= read -r -d '' file; do
         if [[ -e $file || -L $file ]]; then printf '%s\0' "$file"; fi
       done) |
      tar -C "$source" --null --no-recursion -T - -cf - | tar -C "$export_dir/src" -xf -
    if [[ $job == lint ]]; then
      (cd "$source" && "$scripts/lint.sh" --diff) > "$export_dir/lint.diff"
      lint_diff=(-v "$export_dir/lint.diff:/lint.diff:ro" -e AF_LINT_DIFF=/lint.diff)
    fi
    echo "Building a copy of $source without .git; the results stay in $export_dir/src"
    source=$export_dir/src
  fi
  # The repository is mounted, so build trees land in build/ as usual; give them
  # and a ccache directory back to the calling user afterwards. A source tree from
  # elsewhere, and always the GPU jobs' one, is mounted at /src, with this
  # repository read-only at /ci. The GPU tests get the GPU's compute libraries,
  # CUDA's and OpenCL's.
  mounts=(-v "$root:/src")
  script=.github/scripts/ci.sh
  if [[ $job == *-gpu-* || $source != "$root" ]]; then
    mounts=(-v "$root:/ci:ro" -v "$source:/src" -e AF_CI_SOURCE_DIR=/src)
    script=/ci/.github/scripts/ci.sh
    [[ $job == *-test ]] && mounts+=(--gpus all -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility")
  fi
  # shellcheck disable=SC2086
  exec docker run --rm ${AF_CI_DOCKER_ARGS:-} "${mounts[@]}" ${lint_diff[@]+"${lint_diff[@]}"} -w /src \
    -v "$ccache:/ccache" -e CCACHE_DIR=/ccache -e CCACHE_MAXSIZE ${mtx_cache[@]+"${mtx_cache[@]}"} \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/src \
    -e CMAKE_BUILD_PARALLEL_LEVEL -e AF_LINT_BASE -e AF_CI_CUDA_VERSION -e AF_CI_CUDA_ARCHITECTURES \
    -e AF_CI_MTX_SOURCE \
    "$image" bash -c '"$1" "$2"; rc=$?
      chown -R "$3" build docs/details/examples.dox ${4:+/ccache} ${5:+/mtx-cache} 2> /dev/null; exit $rc' \
    bash "$script" "$job" "$(id -u):$(id -g)" "${AF_CI_CCACHE_DIR:-}" "${AF_CI_MTX_CACHE:-}"
fi

if [[ $job == *-gpu-test ]] && ! nvidia-smi; then
  echo "No NVIDIA GPU is visible, and $job needs one (docker run --gpus all)" >&2
  exit 1
fi

if [[ $job == windows-* ]]; then
  export VCPKG_ROOT=${VCPKG_ROOT:-$HOME/vcpkg}
  # The presets' x64-windows-release triplet, from this repository rather than the tree
  # under test, so that commits that predate it build with it too.
  triplets=$(cygpath -m "$root/.github/vcpkg-triplets")
  export VCPKG_OVERLAY_TRIPLETS=$triplets${VCPKG_OVERLAY_TRIPLETS:+;$VCPKG_OVERLAY_TRIPLETS}
fi

if [[ -n $deps ]]; then
  # shellcheck disable=SC2086
  case $job in
    macos-*) "$scripts/install-deps-macos.sh" ;;
    windows-*) pwsh -NoProfile -File "$scripts/install-deps-windows.ps1" ;;
    *) "$scripts/install-deps-ubuntu.sh" $deps_args ;;
  esac
fi

# A source tree from elsewhere gets this repository's presets, which hold the CI
# settings, so commits that predate them can be tested the same way.
if [[ $source != "$root" && $job != *-gpu-* ]]; then
  cp "$root/CMakePresets.json" "$source/CMakePresets.json"
  echo "Building $source with the presets of $root"
fi
cd "$source"

# The tests' Matrix Market data: downloaded from the pinned arrayfire-data commit,
# verified against its manifest and restored into the build tree before the configure,
# or none while no commit is pinned. AF_CI_MTX_FILES, ON or OFF, turns the tests on or
# off in the CI configurations (matrixmarket.cmake and the "ci" preset), which point the
# tests' download at the restored files. Windows' python3 may be a store stub.
case $job in
  lint|docs|workflows|*-gpu-test) ;;
  *)
    python=python3
    "$python" -c '' 2> /dev/null || python=python
    if [[ $job == *-gpu-build ]]; then build_dir=$gpu_build; else build_dir=$source/build/ci-$job; fi
    mtx=()
    [[ -n ${AF_CI_MTX_SOURCE:-} ]] && mtx+=(--source "$AF_CI_MTX_SOURCE")
    [[ -n ${AF_CI_MTX_CACHE:-} ]] && mtx+=(--cache "$AF_CI_MTX_CACHE")
    AF_CI_MTX_FILES=$("$python" "$scripts/check-matrix-data.py" ${mtx[@]+"${mtx[@]}"} "$source" "$build_dir")
    export AF_CI_MTX_FILES=${AF_CI_MTX_FILES%$'\r'}  # Windows' Python ends the line with \r\n
    ;;
esac

# Runs the tests of a *-gpu-build archive, unpacked at the path it was built at.
gpu_tests() {
  [[ -d $gpu_build ]] || tar -C "$source" --zstd -xf "$gpu_build.tar.zst"
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 ctest --test-dir "$gpu_build" \
    --timeout 900 --output-on-failure --no-tests=error \
    --output-junit "$gpu_build/junit.xml" 2>&1 | tee "$gpu_build/ctest.log"
}

case $job in
  lint)
    if [[ -n ${AF_LINT_DIFF:-} ]]; then
      "$scripts/lint.sh" --check "$AF_LINT_DIFF"
    else
      "$scripts/lint.sh"
    fi
    ;;
  linux-icx-mkl|linux-oneapi)
    set +eu
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force > /dev/null
    set -eu
    cmake --workflow --preset "ci-$job"
    ;;
  linux-gcc-openblas-floor)
    # CMake 3.16 cannot read presets; these settings mirror the
    # ci-linux-gcc-openblas preset.
    build=build/ci-$job
    "/opt/cmake-$floor_cmake/bin/cmake" -S . -B "$build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DAF_BUILD_CPU=ON -DAF_BUILD_CUDA=OFF -DAF_BUILD_OPENCL=OFF \
      -DAF_BUILD_ONEAPI=OFF -DAF_BUILD_UNIFIED=OFF -DAF_BUILD_FORGE=ON \
      -DAF_BUILD_EXAMPLES=ON -DAF_BUILD_DOCS=OFF -DAF_COMPUTE_LIBRARY=FFTW/LAPACK/BLAS \
      -C "$scripts/matrixmarket.cmake"
    "/opt/cmake-$floor_cmake/bin/cmake" --build "$build"
    # The distribution's ctest runs the same tests and can write JUnit.
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 ctest --test-dir "$build" \
      -j 2 --output-on-failure --no-tests=error --output-junit "$source/$build/junit.xml"
    ;;
  workflows)
    # actionlint also runs shellcheck on the workflows' run: steps.
    actionlint
    shellcheck .github/scripts/*.sh
    for test in .github/scripts/*-test.py; do
      [[ -e $test ]] || continue  # no Python suites: the pattern stays unexpanded
      python3 "$test"
    done
    ;;
  linux-cuda)
    # The build compiles for an explicit list that includes Blackwell (the
    # ci-linux-cuda preset). check-cuda-release.sh then checks what users and the
    # installers get instead: the default architecture list and the bundled CUDA
    # libraries. Both run, and the job fails if either failed.
    status=0
    cmake --workflow --preset ci-linux-cuda || status=$?
    "$scripts/check-cuda-release.sh" "$source" "$source/build/ci-linux-cuda-release" || status=1
    exit "$status"
    ;;
  windows-cuda)
    cmake --workflow --preset ci-windows-cuda
    # The CUDA DLLs that the installers bundle (AF_INSTALL_STANDALONE=ON, component
    # cuda_dependencies), from the toolkit's bin\ or, from CUDA 13 on, bin\x64\.
    # Only the install rules change, so nothing is rebuilt.
    build=build/ci-windows-cuda
    cmake -S . -B "$build" -DAF_INSTALL_STANDALONE=ON > /dev/null
    prefix=$(mktemp -d) cuda=$(cygpath -u "$CUDA_PATH")
    status=0
    cmake --install "$build" --component cuda_dependencies --prefix "$prefix" || status=1
    for lib in cufft cublas cublasLt cusolver cusparse nvJitLink nvrtc nvrtc-builtins; do
      bundled=$(cd "$prefix" && find . \( -iname "${lib}64_*.dll" -o -iname "${lib}_*.dll" \) | sed 's|^\./||')
      toolkit=$(find "$cuda/bin" -maxdepth 2 -iname "${bundled##*/}" 2> /dev/null | head -n 1)
      if [[ -z $bundled || $bundled == *$'\n'* || -z $toolkit ]]; then
        printf '%-15s %s\n' "$lib" "MISSING or wrong: '${bundled//$'\n'/ }' (toolkit: '${toolkit:-none}')"
        status=1
      else
        printf '%-15s %-32s from %s\n' "$lib" "$bundled" "$toolkit"
      fi
    done
    rm -rf "$prefix"
    exit "$status"
    ;;
  linux-cuda-gpu-build|linux-opencl-gpu-build)
    settings=(-C "$scripts/${job%-build}.cmake")
    [[ $job == linux-cuda-* ]] &&
      settings+=("-DCUDA_architecture_build_targets=${AF_CI_CUDA_ARCHITECTURES:-7.5;8.9}")
    cmake -S "$source" -B "$gpu_build" -G Ninja "${settings[@]}"
    cmake --build "$gpu_build"
    # The test job gets what the tests need, to unpack at the same path: test
    # programs, shared libraries, CTest files and test data.
    (cd "$source" && find "${gpu_build#"$source/"}" \( -name CMakeFiles -o -name .git \) -prune -o \
      \( -name CTestTestfile.cmake -o -name '*.so*' -o -path '*/test/*' -type f -perm -u+x \
         -o -path '*/extern/af_test_data-src/*' -o -path '*/extern/matrixmarket/*' \) \
      \( -type f -o -type l \) -print) |
      tar -C "$source" --zstd -cf "$gpu_build.tar.zst" -T -
    ls -l "$gpu_build.tar.zst"
    ;;
  linux-cuda-gpu-test)
    # A driver older than the toolkit can't load the PTX that ArrayFire's JIT makes;
    # NVIDIA's forward-compatibility package brings a newer CUDA driver library for
    # data-center GPUs such as the T4.
    compat=$(nvidia-smi | "$scripts/cuda-compat.sh" "$cuda_version")
    read -r mode driver_cuda <<< "$compat"
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    if [[ $mode == compat ]]; then
      echo "Driver $driver supports CUDA $driver_cuda, older than CUDA $cuda_version:" \
        "compat, with NVIDIA's forward-compatibility package cuda-compat-${cuda_version//./-}"
      apt-get install -y -q --no-install-recommends "cuda-compat-${cuda_version//./-}"
      export LD_LIBRARY_PATH=/usr/local/cuda-$cuda_version/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    else
      echo "Driver $driver supports CUDA $driver_cuda, enough for CUDA $cuda_version: native"
    fi
    gpu_tests
    ;;
  linux-opencl-gpu-test)
    # The container toolkit mounts the driver's OpenCL library with the compute
    # capability, but only its versions from mid-2026 on also mount the ICD file
    # through which the loader finds it.
    icd=/etc/OpenCL/vendors/nvidia.icd
    if ! grep -qs libnvidia-opencl.so.1 "$icd"; then
      mkdir -p "${icd%/*}"
      echo libnvidia-opencl.so.1 > "$icd"
      echo "Registered libnvidia-opencl.so.1 in $icd"
    fi
    platforms=$(clinfo -l || true)
    echo "OpenCL platforms (clinfo -l):"
    echo "${platforms:-  none}"
    if ! grep -q '^Platform #[0-9]*: NVIDIA' <<< "$platforms"; then
      grep -q libnvidia-opencl.so.1 <<< "$(ldconfig -p)" ||
        echo "The driver's OpenCL library libnvidia-opencl.so.1 isn't in the container." >&2
      echo "No NVIDIA OpenCL platform, so the OpenCL tests can't use the GPU" >&2
      exit 1
    fi
    AF_OPENCL_DEVICE_TYPE=GPU gpu_tests
    ;;
  *)
    cmake --workflow --preset "ci-$job"
    ;;
esac
