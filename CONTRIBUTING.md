# Contributing to ArrayFire #

## Reporting Issues ##

We both value and encourage the community to provide feedback about using ArrayFire and the issues they are facing.
The more detailed the information, the easier it is for the developers to resolve the issue.

Issues can span a variety of topics including:
- Feature requests
- Bug reports
- Build Issues
- Performance improvements
- New hardware / backend support

We use the github [issue tracker](https://github.com/arrayfire/arrayfire/issues?state=open) to track our issues. Technical issues can also be discussed on our [user forum](https://groups.google.com/forum/#!forum/arrayfire-users).

## Contributing Code ##

If you want to contribute code, we suggest you use the one of the following methods.

- [Using Github](https://github.com/arrayfire/arrayfire/wiki/Contribute-code-using-github)
- [Using Email](https://github.com/arrayfire/arrayfire/wiki/Contribute-code-using-email)

Key areas of code contributions include:
- [New features](https://github.com/arrayfire/arrayfire/issues?q=is%3Aopen+is%3Aissue+label%3Afeature)
    - You can refer to [this](https://github.com/arrayfire/arrayfire/wiki/How-to-Add-a-New-Function-to-ArrayFire) wiki page to get started on how to add a new function to the library.
- [Bug fixes](https://github.com/arrayfire/arrayfire/labels/bug)
- [Style changes](https://github.com/arrayfire/arrayfire/labels/style)
- [Performance improvements](https://github.com/arrayfire/arrayfire/labels/style)
- [New tests](https://github.com/arrayfire/arrayfire/labels/test)
- New examples!

## Running CI ##

CI runs in three tiers:

- **Every push** to a pull request or to `master` runs the light checks: `lint` (formatting of
  the changed lines), `docs`, and `workflows` (actionlint and shellcheck over the workflows and
  CI scripts).
- **The CPU tests** (Linux with GCC and Clang, macOS and Windows) run in the merge queue and on
  every push to `master`. A pull request shows them as passing at once ("runs in the merge
  queue") so that it can be queued. A merge-queue entry or a push to `master` that only adds
  or changes non-code files builds nothing: the four checks pass in seconds and say why. The
  non-code files are Markdown files at the top level, `docs/` (which the `docs` check
  builds), the issue and pull-request templates, `.github/CITATION.md`, `CITATION.cff`,
  `.gitignore` and `.mailmap`. Any other file, or removing or renaming a file, runs the
  tests in full, and so does every request. To run them on the pull request's current
  head, comment `/cpu-tests`, or `/cpu-tests all` to add the compile-only accelerator builds:
  - `accelerators (linux-cuda, <version>)` for each CUDA version in `AF_CI_CUDA_VERSIONS`
    (default 12.9). It builds the CUDA backend for compute capabilities 7.5, 8.9 and 9.0 and
    for Blackwell's 10.0 and 12.0. Then it configures the backend as users and the
    installers get it: it compiles one CUDA file with ArrayFire's default architecture
    list, and checks that every CUDA library an `AF_INSTALL_STANDALONE` installer bundles is
    there under the name ArrayFire loads it by.
  - `windows-cuda (<version>)`, the same build with MSVC for compute capability 7.5, and
    the check of the bundled DLLs;
  - `accelerators (linux-opencl)` and `accelerators (linux-oneapi)`.

  Maintainers with write access and the Claude bot can ask on any pull request. The pull
  request's author can ask on their own pull request once the light checks have run on it.
  Requests by anyone other than a maintainer run the Linux jobs only. The result appears as
  the `cpu-tests` status and a comment.
- **The GPU tests** run only on request: a maintainer with write access comments
  `/gpu-tests [cuda|opencl|all] [cuda=<major.minor>]`.
  - `/gpu-tests` alone lets the pull request's changed files choose. Changes to
    `src/backend/cuda/`, `include/af/cuda.h`, `test/cuda.cu` or CUDA's CMake helpers test CUDA;
    changes to `src/backend/opencl/`, `include/af/opencl.h`, the OpenCL interop tests or
    OpenCL's CMake helpers test OpenCL. Anything shared, such as `src/backend/common/`,
    `src/api/`, `include/`, `test/`, CMake files and `vcpkg.json`, tests both. Markdown,
    `docs/`, `.github/`, the CPU and oneAPI backends and `examples/` test nothing: the reply
    says so and nothing starts. `/gpu-tests all` runs the tests anyway.
  - `/gpu-tests cuda`, `/gpu-tests opencl` and `/gpu-tests all` choose the backends directly.
  - CUDA is tested once for each version in the repository variable `AF_CI_CUDA_VERSIONS`
    (for example `12.9 13.4`), or with CUDA 12.9 when it is not set. `cuda=13.4` tests
    that one version instead: `/gpu-tests cuda=13.4` tests CUDA 13.4 only, and
    `/gpu-tests all cuda=13.4` tests CUDA 13.4 and OpenCL.

  The reply says what runs and which changed file chose it. Each CUDA version, and OpenCL,
  is a separate leg: it is built on a standard runner and then tested on its own GPU
  runner, so a failed build stops only its own leg. One approval in the `gpu-tests`
  environment covers every leg of a request, and a maintainer can give it while the builds
  run. The result appears as the `gpu-tests` status and a comment that lists each leg's
  passed and failed tests. A GPU runner (an NVIDIA T4) costs about 5 cents a minute, and
  each leg uses it for roughly 10 to 15 minutes. A plain `/gpu-tests` on a CUDA change
  therefore costs well under a dollar, and `/gpu-tests all` with two CUDA versions costs
  about three times that. The builds run on standard runners, which cost nothing for a
  public repository.

Requests test the exact commit that is the pull request's head at that moment, with the CI
scripts from `master`. When the pull request's branch is behind `master`, the reply says by how
many commits: CI fixes made on `master` since then reach the test only after "Update branch".
Maintainers can also start the "CPU tests" or "GPU tests" workflow by
hand with a commit SHA. The GPU tests are off until the repository variable `AF_CI_GPU_TESTS`
is `true` and `AF_CI_GPU_RUNNER` names the GPU runners' label. `/cpu-tests` also uses the CMake presets from `master`, while the merge
queue uses the pull request's own, so a change to `CMakePresets.json` is really tested in the
merge queue.

## Running the CI Checks Locally ##

Each check that CI runs is a job of `.github/scripts/ci.sh` (`--list` shows them). With
Docker, a Linux job runs in the same Ubuntu image that CI uses:

    .github/scripts/ci.sh --docker lint
    .github/scripts/ci.sh --docker docs
    .github/scripts/ci.sh --docker workflows
    .github/scripts/ci.sh --docker linux-gcc-openblas

To test another checkout with this one's scripts and CI settings, set `AF_CI_SOURCE_DIR` to
its directory.

The sparse tests read Matrix Market files from the SuiteSparse collection, which refuses CI's
downloads. `ci.sh` checks the copies bundled in `.github/fixtures` and restores them into the
build tree before every build, and the CI presets point the tests' download at them. To build
a `ci-*` preset without `ci.sh`, run
`python3 .github/scripts/check-matrix-data.py . build/<preset>` first.

The build tree goes to `build/<preset>` and test results to `build/<preset>/junit.xml`. With
`--docker`, `ci.sh` picks the build parallelism from the memory the container can use and
prints its choice; set `CMAKE_BUILD_PARALLEL_LEVEL` to choose it yourself.

Without `--docker`, `ci.sh` first installs the job's packages with apt, Homebrew or vcpkg;
`--no-deps` skips that. The build itself is a CMake workflow preset, which needs CMake 3.25 or
newer, for example:

    cmake --workflow --preset ci-linux-gcc-openblas

`lint` checks the formatting of the lines you changed since the default branch with
clang-format 15.0.7.

##ArrayFire Based Projects##

You can also contribute to ArrayFire by helping out projects that use ArrayFire! For our part, in addition to the ArrayFire library we are also in the process of adding native bindings for numerous language. We currently support

- [Java](https://github.com/arrayfire/arrayfire_java)
- [R](https://github.com/arrayfire/arrayfire_r)
- [Fortran](https://github.com/arrayfire/arrayfire_fortran)

If you are experienced in any of these languages, you can help us improve these language bindings. If you prefer a different language that is not in the list, dive in and create a new repo!
