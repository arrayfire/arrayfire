#!/usr/bin/env python3
"""Fixture tests of the "/cpu-tests" and "/gpu-tests" decisions in test-requests.py, of how
its commands handle failed API calls and report GPU runs, and of cuda-compat.sh's choice
between the host's driver and NVIDIA's forward-compatibility package.

Each decision case is an issue_comment event plus the API answers; the test prints the
expected and the actual decision and fails if any case differs. The command cases run
test-requests.py as the workflow's jobs do, on fixtures, with some writes failing.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest.mock
import urllib.error
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-requests.py")
CUDA_COMPAT = os.path.join(os.path.dirname(SCRIPT), "cuda-compat.sh")
spec = importlib.util.spec_from_file_location("test_requests", SCRIPT)
test_requests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_requests)

HEAD = "cd211d5b8660761d1665be52ace5ff680856ba06"
# An earlier head of the pull request, green on every check; it must not count.
STALE_HEAD = "0" * 40
PR = 3715
SETTINGS = {"gpu_tests": "true", "gpu_runner": "gpu-t4", "accelerator_builds": "true", "cuda_versions": ""}
ASSOCIATION = {"maintainer-a": "MEMBER", "admin-b": "OWNER", "writer-c": "COLLABORATOR",
               "reader-d": "COLLABORATOR", "triager-e": "COLLABORATOR",
               "contributor-g": "CONTRIBUTOR", "github-actions[bot]": "NONE"}


def event(login, body="/gpu-tests", author="maintainer-a", association=None, pull_request=True):
    issue = {"number": PR, "title": "Add support for CUDA 13.1", "user": {"login": author}}
    if pull_request:
        issue["pull_request"] = {"url": f"https://api.github.com/repos/arrayfire/arrayfire/pulls/{PR}"}
    return {"action": "created", "issue": issue,
            "comment": {"body": body, "user": {"login": login},
                        "author_association": association or ASSOCIATION.get(login, "NONE")}}


def checks(app="github-actions", **conclusions):
    """The light checks on a commit; a conclusion of None means still running."""
    names = ("lint", "docs", "workflows")
    return [{"id": 10 + i, "name": name, "app": {"slug": app},
             "conclusion": conclusions.get(name, "success")} for i, name in enumerate(names)]


def changed(*paths, renamed_from=None):
    """The pull request's changed files, as GitHub lists them."""
    files = [{"filename": path, "status": "modified"} for path in paths]
    if renamed_from:
        files[0].update(status="renamed", previous_filename=renamed_from)
    return files


CUDA_FILE = "src/backend/cuda/jit.cpp"
OPENCL_FILE = "src/backend/opencl/jit.cpp"
SHARED_HEADER = "include/af/array.h"
DOCS = ("docs/pages/install.md", "README.md")


def answers(author="maintainer-a", check_runs=None, files=None):
    """API answers; by default the light checks passed on HEAD, and the pull request changes
    one file of the CUDA backend."""
    return {"permissions": {"maintainer-a": "maintain", "admin-b": "admin", "writer-c": "write",
                            "reader-d": "read", "triager-e": "triage"},
            "pull": {"number": PR, "head": {"sha": HEAD}, "base": {"ref": "master"}, "user": {"login": author}},
            "check_runs": {HEAD: checks() if check_runs is None else check_runs,
                           STALE_HEAD: checks()},
            "files": changed(CUDA_FILE) if files is None else files}


def accept(command, backends="", cuda_versions="", accelerators=False, macos_windows=False):
    return ("accept", "accepted", command, HEAD, backends, cuda_versions, accelerators, macos_windows)


def refuse(command, reason):
    return ("reply", reason, command, "", "", "", False, False)


def ignore(reason, command=""):
    return ("ignore", reason, command, "", "", "", False, False)


GPU_CALLS = ["permission maintainer-a", f"pull {PR}", f"files {PR}"]
# An accepted request also asks how far the head is behind its base branch.
COMPARE = f"compare master...{HEAD}"
GPU_ACCEPTED = GPU_CALLS + [COMPARE]


CASES = [
    # name, event, answers, settings changes, expected, expected calls
    # /cpu-tests: writers
    ("cpu: maintainer is accepted, with macOS and Windows", event("maintainer-a", "/cpu-tests"),
     answers(), {}, accept("cpu", macos_windows=True), ["permission maintainer-a", f"pull {PR}", COMPARE]),
    ("cpu: writer's /cpu-tests all adds the accelerator builds", event("writer-c", "/cpu-tests all"),
     answers(), {}, accept("cpu", accelerators=True, macos_windows=True), None),
    ("cpu: /cpu-tests all says when accelerator builds are disabled", event("writer-c", "/cpu-tests all"),
     answers(), {"accelerator_builds": ""}, accept("cpu", accelerators=True, macos_windows=True), None),
    # /cpu-tests: the pull request's own author without write access
    ("cpu: author without write access, light checks done, is accepted on Linux",
     event("contributor-g", "/cpu-tests", author="contributor-g"), answers(author="contributor-g"), {},
     accept("cpu"), ["permission contributor-g", f"pull {PR}", f"check-runs {HEAD}", COMPARE]),
    ("cpu: first-time author whose light checks never ran is refused",
     event("newcomer-h", "/cpu-tests", author="newcomer-h", association="FIRST_TIME_CONTRIBUTOR"),
     answers(author="newcomer-h", check_runs=[]), {}, refuse("cpu", "light-checks-pending"), None),
    ("cpu: author is refused while a light check is still running",
     event("contributor-g", "/cpu-tests", author="contributor-g"),
     answers(author="contributor-g", check_runs=checks(docs=None)), {},
     refuse("cpu", "light-checks-pending"), None),
    ("cpu: light checks from another app don't count",
     event("contributor-g", "/cpu-tests", author="contributor-g"),
     answers(author="contributor-g", check_runs=checks(app="some-other-app")), {},
     refuse("cpu", "light-checks-pending"), None),
    ("cpu: someone else's pull request is not the author's", event("reader-d", "/cpu-tests"),
     answers(), {}, refuse("cpu", "not-allowed"), None),
    ("cpu: stranger is ignored silently", event("contributor-g", "/cpu-tests"), answers(), {},
     ignore("stranger", "cpu"), []),
    ("cpu: stranger's malformed request gets no usage reply", event("contributor-g", "/cpu-tests now"),
     answers(), {}, ignore("stranger", "cpu"), []),
    ("cpu: malformed request gets the usage", event("maintainer-a", "/cpu-tests everything"),
     answers(), {}, refuse("cpu", "usage"), []),

    # /gpu-tests: writers
    ("gpu: maintainer is accepted", event("maintainer-a"), answers(), {}, accept("gpu", "cuda", "12.9"), None),
    ("gpu: admin with cuda=13.1 is accepted", event("admin-b", "/gpu-tests cuda=13.1\nthanks"),
     answers(), {}, accept("gpu", "cuda", "13.1"), None),
    # /gpu-tests: the author without write access, and other non-writers
    ("gpu: author without write access is refused",
     event("contributor-g", author="contributor-g"), answers(author="contributor-g"), {},
     refuse("gpu", "not-allowed"), ["permission contributor-g"]),
    ("gpu: non-writer is refused", event("reader-d"), answers(), {}, refuse("gpu", "not-allowed"),
     ["permission reader-d"]),
    ("gpu: triage role is refused", event("triager-e"), answers(), {}, refuse("gpu", "not-allowed"), None),
    ("gpu: stranger is ignored silently", event("contributor-g"), answers(), {},
     ignore("stranger", "gpu"), []),
    ("gpu: gate off answers disabled without any API call", event("maintainer-a"), answers(),
     {"gpu_tests": ""}, refuse("gpu", "disabled"), []),
    ("gpu: no GPU runner configured is refused up front", event("maintainer-a"), answers(),
     {"gpu_runner": ""}, refuse("gpu", "no-runner"), []),
    ("gpu: malformed cuda argument gets the usage", event("maintainer-a", "/gpu-tests cuda=thirteen"),
     answers(), {}, refuse("gpu", "usage"), []),
    ("gpu: unknown argument gets the usage", event("maintainer-a", "/gpu-tests now please"), answers(),
     {}, refuse("gpu", "usage"), []),
    ("gpu: `opencl` with a CUDA version gets the usage", event("maintainer-a", "/gpu-tests opencl cuda=13.4"),
     answers(), {}, refuse("gpu", "usage"), []),
    ("gpu: an unknown backend gets the usage", event("maintainer-a", "/gpu-tests both"), answers(), {},
     refuse("gpu", "usage"), []),
    ("gpu: a backend in capitals gets the usage", event("maintainer-a", "/gpu-tests CUDA"), answers(), {},
     refuse("gpu", "usage"), []),
    ("gpu: a major-only version gets the usage", event("maintainer-a", "/gpu-tests cuda=13"), answers(), {},
     refuse("gpu", "usage"), []),
    ("gpu: the version before the backend gets the usage", event("maintainer-a", "/gpu-tests cuda=13.4 all"),
     answers(), {}, refuse("gpu", "usage"), []),
    ("gpu: two backends get the usage", event("maintainer-a", "/gpu-tests cuda opencl"), answers(), {},
     refuse("gpu", "usage"), []),
    ("gpu: two versions get the usage", event("maintainer-a", "/gpu-tests cuda=12.9 cuda=13.4"), answers(),
     {}, refuse("gpu", "usage"), []),

    # /gpu-tests forms: an explicit backend or version runs whatever the files are
    ("gpu form: `/gpu-tests cuda` runs CUDA with the default versions", event("maintainer-a", "/gpu-tests cuda"),
     answers(files=changed(OPENCL_FILE)), {}, accept("gpu", "cuda", "12.9"), GPU_ACCEPTED),
    ("gpu form: `/gpu-tests opencl` runs OpenCL only", event("maintainer-a", "/gpu-tests opencl"),
     answers(), {}, accept("gpu", "opencl"), GPU_ACCEPTED),
    ("gpu form: `/gpu-tests all` runs CUDA and OpenCL", event("maintainer-a", "/gpu-tests all"),
     answers(), {}, accept("gpu", "cuda opencl", "12.9"), GPU_ACCEPTED),
    ("gpu form: `/gpu-tests cuda=13.4` runs CUDA 13.4 only", event("maintainer-a", "/gpu-tests cuda=13.4"),
     answers(files=changed(SHARED_HEADER)), {"cuda_versions": "12.9 13.4"}, accept("gpu", "cuda", "13.4"),
     GPU_ACCEPTED),
    ("gpu form: `/gpu-tests cuda cuda=13.4` runs CUDA 13.4", event("maintainer-a", "/gpu-tests cuda cuda=13.4"),
     answers(), {}, accept("gpu", "cuda", "13.4"), GPU_ACCEPTED),
    ("gpu form: `/gpu-tests all cuda=13.4` runs CUDA 13.4 and OpenCL",
     event("maintainer-a", "/gpu-tests all cuda=13.4"), answers(), {"cuda_versions": "12.9"},
     accept("gpu", "cuda opencl", "13.4"), GPU_ACCEPTED),
    ("gpu form: an explicit request runs on a docs-only pull request",
     event("maintainer-a", "/gpu-tests all"), answers(files=changed(*DOCS)), {},
     accept("gpu", "cuda opencl", "12.9"), GPU_ACCEPTED),
    ("gpu form: extra spaces between the words are fine", event("maintainer-a", "/gpu-tests   all   cuda=13.4  "),
     answers(), {}, accept("gpu", "cuda opencl", "13.4"), GPU_ACCEPTED),

    # /gpu-tests without a backend: the changed files choose
    ("gpu files: a docs-only change gets a reply and starts nothing", event("maintainer-a"),
     answers(files=changed(*DOCS)), {}, refuse("gpu", "no-gpu-changes"), GPU_CALLS),
    ("gpu files: CI, the CPU and oneAPI backends and the examples start nothing", event("maintainer-a"),
     answers(files=changed(".github/workflows/gpu.yml", "src/backend/cpu/jit.cpp", "src/backend/oneapi/jit.cpp",
                           "examples/helloworld/helloworld.cpp", "include/af/oneapi.h", "LICENSE")), {},
     refuse("gpu", "no-gpu-changes"), GPU_CALLS),
    ("gpu files: a pull request without changed files starts nothing", event("maintainer-a"),
     answers(files=[]), {}, refuse("gpu", "no-gpu-changes"), GPU_CALLS),
    ("gpu files: a CUDA file runs CUDA with the default versions", event("maintainer-a"),
     answers(files=changed(CUDA_FILE)), {}, accept("gpu", "cuda", "12.9"), GPU_ACCEPTED),
    ("gpu files: an OpenCL file runs OpenCL", event("maintainer-a"), answers(files=changed(OPENCL_FILE)), {},
     accept("gpu", "opencl"), GPU_ACCEPTED),
    ("gpu files: a shared header runs both", event("maintainer-a"), answers(files=changed(SHARED_HEADER)), {},
     accept("gpu", "cuda opencl", "12.9"), GPU_ACCEPTED),
    ("gpu files: a mixed list runs both, each for its file", event("maintainer-a"),
     answers(files=changed(DOCS[0], OPENCL_FILE, ".github/workflows/gpu.yml", CUDA_FILE)), {},
     accept("gpu", "cuda opencl", "12.9"), GPU_ACCEPTED),
    ("gpu files: CUDA's tests and CMake helpers run CUDA", event("maintainer-a"),
     answers(files=changed("test/cuda.cu", "CMakeModules/select_compute_arch.cmake", "include/af/cuda.h")), {},
     accept("gpu", "cuda", "12.9"), GPU_ACCEPTED),
    ("gpu files: OpenCL's tests and CMake helpers run OpenCL", event("maintainer-a"),
     answers(files=changed("test/ocl_ext_context.cpp", "CMakeModules/build_clFFT.cmake")), {},
     accept("gpu", "opencl"), GPU_ACCEPTED),
    ("gpu files: shared backend code, the APIs, CMake and manifests run both", event("maintainer-a"),
     answers(files=changed("src/backend/common/jit/Node.cpp", "src/api/c/data.cpp", "CMakeLists.txt",
                           "vcpkg.json", "test/arrayfire_test.cpp")), {},
     accept("gpu", "cuda opencl", "12.9"), GPU_ACCEPTED),
    ("gpu files: a source file renamed out of the CUDA backend runs CUDA", event("maintainer-a"),
     answers(files=changed("src/backend/cpu/moved.cpp", renamed_from="src/backend/cuda/moved.cpp")), {},
     accept("gpu", "cuda", "12.9"), GPU_ACCEPTED),
    ("gpu files: Markdown in a backend directory starts nothing", event("maintainer-a"),
     answers(files=changed("src/backend/cuda/README.md")), {}, refuse("gpu", "no-gpu-changes"), GPU_CALLS),

    # CUDA versions: AF_CI_CUDA_VERSIONS, or 12.9 without it
    ("gpu versions: AF_CI_CUDA_VERSIONS lists every version to run", event("maintainer-a"),
     answers(), {"cuda_versions": "12.9 13.4"}, accept("gpu", "cuda", "12.9 13.4"), GPU_ACCEPTED),
    ("gpu versions: AF_CI_CUDA_VERSIONS with both backends", event("maintainer-a", "/gpu-tests all"),
     answers(), {"cuda_versions": "12.9 13.4"}, accept("gpu", "cuda opencl", "12.9 13.4"), GPU_ACCEPTED),
    ("gpu versions: repeats and extra spaces in AF_CI_CUDA_VERSIONS are dropped", event("maintainer-a"),
     answers(), {"cuda_versions": " 13.4  12.9 13.4 "}, accept("gpu", "cuda", "13.4 12.9"), GPU_ACCEPTED),
    ("gpu versions: a blank AF_CI_CUDA_VERSIONS means 12.9", event("maintainer-a"),
     answers(), {"cuda_versions": "  "}, accept("gpu", "cuda", "12.9"), GPU_ACCEPTED),
    ("gpu versions: a malformed AF_CI_CUDA_VERSIONS is refused", event("maintainer-a"),
     answers(), {"cuda_versions": "12.9,13.4"}, refuse("gpu", "bad-cuda-versions"), GPU_CALLS),
    ("gpu versions: OpenCL alone doesn't need AF_CI_CUDA_VERSIONS", event("maintainer-a", "/gpu-tests opencl"),
     answers(), {"cuda_versions": "latest"}, accept("gpu", "opencl"), GPU_ACCEPTED),

    # Comments that are not requests
    ("github-actions[bot] comment is ignored", event("github-actions[bot]"), answers(), {},
     ignore("own-comment"), []),
    ("comment on an issue is ignored", event("maintainer-a", pull_request=False), answers(), {},
     ignore("not-pull-request"), []),
    ("command not on the first line is ignored", event("maintainer-a", "please run\n/gpu-tests"),
     answers(), {}, ignore("not-a-request"), []),
    ("a longer word is not the command", event("maintainer-a", "/gpu-testsuite"), answers(), {},
     ignore("not-a-request"), []),
]


BEHIND = "This branch is"
BEHIND_CASES = [
    # name, event, answers changes, the line expected in the reply (None: no line), warning
    ("behind: a CPU request on a branch 3 commits behind master gets the update line",
     event("maintainer-a", "/cpu-tests"), {"behind_by": 3},
     "This branch is 3 commits behind `master`; CI fixes made there since aren't in this test. "
     "Use \"Update branch\" (or merge `master`) to include them.", False),
    ("behind: a GPU request on a branch 1 commit behind master says \"1 commit\"",
     event("maintainer-a", "/gpu-tests"), {"behind_by": 1}, "This branch is 1 commit behind `master`;", False),
    ("behind: an up-to-date branch gets no line", event("maintainer-a", "/gpu-tests"), {"behind_by": 0},
     None, False),
    ("behind: a failing compare call is a warning, and the request is still accepted",
     event("maintainer-a", "/cpu-tests"),
     {"behind_by": 3, "fail": {"compare": {"status": 404, "body": {"message": "Not Found"}}}}, None, True),
]


def behind_case(name, ev, changes, line, warning):
    api = test_requests.Fixtures({**answers(), **changes})
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = test_requests.decide(ev, SETTINGS, api)
    printed = output.getvalue()
    warned = printed.startswith(f"::warning::GitHub API GET compare/master...{HEAD} failed: HTTP 404: Not Found")
    ok = (result["action"] == "accept" and f"compare master...{HEAD}" in api.calls
          and (line in result["reply"] if line else BEHIND not in result["reply"])
          and warned == warning and "::error::" not in printed)
    print(f"{'ok  ' if ok else 'FAIL'} {name}\n"
          f"     decision: {result['action']} ({result['reason']}), API calls: {api.calls}"
          + (f"\n     printed: {printed.strip()}" if printed.strip() else "")
          + f"\n     reply: {result['reply']}")
    return ok


def api_error_annotation():
    """GitHub.call turns an HTTP error into an annotation with the method, the path, the
    status and GitHub's message, cut to about 200 characters, and re-raises it; the request's
    headers, which hold the token, are not in it."""
    token = "test-token-not-a-credential"
    sent = []

    def urlopen(request, timeout):
        sent.append(request)
        body = {"message": "Resource not accessible by integration " + "x" * 300,
                "documentation_url": "https://docs.github.com/rest/issues/comments"}
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {"X-GitHub-Request-Id": "1"},
                                     io.BytesIO(json.dumps(body).encode()))

    output = io.StringIO()
    with unittest.mock.patch.object(urllib.request, "urlopen", urlopen), \
            contextlib.redirect_stdout(output):
        try:
            test_requests.GitHub("arrayfire/arrayfire", token).comment(PR, "CPU tests requested.")
            raised = None
        except urllib.error.HTTPError as error:
            raised = error.code
    printed = output.getvalue()
    prefix = f"::error::GitHub API POST issues/{PR}/comments failed: HTTP 403: "
    message = printed.strip()[len(prefix):]
    actual = (raised, sent[0].get_header("Authorization") == f"Bearer {token}",
              printed.startswith(prefix) and printed.count("\n") == 1,
              message.startswith("Resource not accessible by integration") and len(message) <= 200,
              "Authorization" in printed or "Bearer" in printed or token in printed)
    expected = (403, True, True, True, False)
    print(f"{'ok  ' if actual == expected else 'FAIL'} an API error is annotated without the request's headers\n"
          f"     expected: re-raised, token sent, one annotation, message cut to 200, headers shown = {expected}\n"
          f"     actual:   re-raised, token sent, one annotation, message cut to 200, headers shown = {actual}\n"
          f"     printed: {printed.strip()}")
    return actual == expected


def permission_not_found():
    """permission() still reads a 404 as no access, without an annotation."""
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {},
                                     io.BytesIO(b'{"message": "Not Found"}'))

    output = io.StringIO()
    with unittest.mock.patch.object(urllib.request, "urlopen", urlopen), \
            contextlib.redirect_stdout(output):
        role = test_requests.GitHub("arrayfire/arrayfire", "").permission("contributor-g")
    actual = (role, output.getvalue())
    expected = ("none", "")
    print(f"{'ok  ' if actual == expected else 'FAIL'} permission() reads a 404 as no access, quietly\n"
          f"     expected: role, output = {expected}\n"
          f"     actual:   role, output = {actual}")
    return actual == expected


def run_command(command, fail, env=None):
    """Runs test-requests.py <command> on fixtures, as the workflow's jobs run it; fail maps a
    write (comment, set_status) to the HTTP status and message it fails with, and env replaces
    the jobs' environment variables."""
    fixtures = {"fail": {name: {"status": code, "body": {"message": message}}
                         for name, (code, message) in fail.items()}}
    with tempfile.TemporaryDirectory() as tmp:
        files = {}
        for name, value in (("event", event("maintainer-a", "/cpu-tests")), ("fixtures", fixtures)):
            files[name] = os.path.join(tmp, f"{name}.json")
            with open(files[name], "w") as file:
                json.dump(value, file)
        process = subprocess.run(
            [sys.executable, "-I", SCRIPT, command], capture_output=True, text=True, timeout=60,
            env={"PATH": os.environ.get("PATH", ""), "ACT": "true",
                 "AF_CI_REQUEST_FIXTURES": files["fixtures"], "GITHUB_EVENT_PATH": files["event"],
                 "GITHUB_REPOSITORY": "arrayfire/arrayfire", "GITHUB_RUN_ID": "900",
                 "COMMAND": "cpu", "SHA": HEAD, "BACKENDS": "", "CUDA_VERSIONS": "",
                 "REPLY": "CPU tests requested.", "RESULT": "success", "SUMMARY": "", "REPORT": "",
                 **(env or {})})
    return process.returncode, process.stdout + process.stderr


FORBIDDEN = (403, "Resource not accessible by integration")
COMMENT_FAILED = f"GitHub API POST issues/{PR}/comments failed: HTTP 403: Resource not accessible by integration"
STATUS_FAILED = f"GitHub API POST statuses/{HEAD} failed: HTTP 403: Resource not accessible by integration"
GPU_RUN = {"COMMAND": "gpu", "BACKENDS": "cuda opencl", "CUDA_VERSIONS": "12.9 13.4",
           "REPLY": "GPU tests requested."}
GPU_SUMMARY = "CUDA 12.9: 132 of 132 passed; CUDA 13.4: build failed; OpenCL: 126 of 128 passed"
GPU_REPORT = ("- CUDA 12.9: 132 of 132 passed\n- CUDA 13.4: build failed\n"
              "- OpenCL: 126 of 128 passed; failed: test_jit_opencl, test_fft_opencl")
COMMAND_CASES = [
    # name, command, failing writes, exits with 0, printed, not printed[, environment]
    ("accepted: sets the pending status and comments", "accepted", {}, True,
     [f"[fixture] status cpu-tests=pending on {HEAD}", f"[fixture] comment on #{PR}"], ["::"]),
    ("accepted: a failed comment is a warning; the status stays and the tests go on", "accepted",
     {"comment": FORBIDDEN}, True,
     [f"[fixture] status cpu-tests=pending on {HEAD}", f"::warning::{COMMENT_FAILED}"],
     ["[fixture] comment", "::error::"]),
    ("accepted: a failed status fails the job", "accepted", {"set_status": FORBIDDEN}, False,
     [f"::error::{STATUS_FAILED}"], ["[fixture] status", "[fixture] comment"]),
    ("report: a failed comment is a warning; the status stays", "report", {"comment": FORBIDDEN}, True,
     [f"[fixture] status cpu-tests=success on {HEAD}", f"::warning::{COMMENT_FAILED}"],
     ["[fixture] comment", "::error::"]),
    ("report: a failed status fails the job", "report", {"set_status": FORBIDDEN}, False,
     [f"::error::{STATUS_FAILED}"], ["[fixture] status", "[fixture] comment"]),
    ("reply: comments", "reply", {}, True, [f"[fixture] comment on #{PR}:\nCPU tests requested."], ["::"]),
    ("reply: a failed comment fails the job", "reply", {"comment": FORBIDDEN}, False,
     [f"::error::{COMMENT_FAILED}"], ["[fixture] comment", "::warning::"]),
    ("accepted: the GPU status names every leg", "accepted", {}, True,
     [f"[fixture] status gpu-tests=pending on {HEAD}: GPU tests (CUDA 12.9, CUDA 13.4 and OpenCL): "
      "building, then waiting for a maintainer's approval"], ["::"], GPU_RUN),
    ("report: every GPU leg is in the status and the comment", "report", {}, True,
     [f"[fixture] status gpu-tests=failure on {HEAD}: GPU tests (CUDA 12.9, CUDA 13.4 and OpenCL) failed: "
      "CUDA 12.9: 132 of 132 passed; CUDA 13.4: build failed; OpenCL: 126 of 128",
      f"[fixture] comment on #{PR}:\nGPU tests (CUDA 12.9, CUDA 13.4 and OpenCL) of `{HEAD}` failed: "
      f"{GPU_SUMMARY} [Run](", f"\n\n{GPU_REPORT}"], ["::"],
     {**GPU_RUN, "RESULT": "failure", "SUMMARY": GPU_SUMMARY, "REPORT": GPU_REPORT}),
    ("report: GPU jobs that were all skipped are an error", "report", {}, True,
     [f"[fixture] status gpu-tests=error on {HEAD}: GPU tests (CUDA 12.9) did not run: gpu.yml's jobs "
      "were skipped"], ["::"], {**GPU_RUN, "BACKENDS": "cuda", "CUDA_VERSIONS": "12.9"}),
]


def command_case(name, command, fail, succeeds, printed, not_printed, env=None):
    code, output = run_command(command, fail, env)
    ok = ((code == 0) == succeeds and all(text in output for text in printed)
          and not any(text in output for text in not_printed))
    shown = [line for line in output.splitlines()
             if line.startswith(("::", "[fixture]", "Not commenting", "- "))]
    print(f"{'ok  ' if ok else 'FAIL'} {name}\n"
          f"     expected: exit {'0' if succeeds else 'non-zero'}\n"
          f"     actual:   exit {code}"
          + "".join(f"\n     printed: {line}" for line in shown))
    return ok


# nvidia-smi's header with the CUDA version the driver supports: "CUDA Version" before
# driver 600, "CUDA UMD Version" from then on (this one is the RTX 4070 Ti's under WSL).
SMI_12_8 = "| NVIDIA-SMI 570.124.06   Driver Version: 570.124.06   CUDA Version: 12.8     |"
SMI_13_4 = "| NVIDIA-SMI 615.78.02    KMD Version: 617.14    CUDA UMD Version: 13.4     |"
SMI_13_6 = "| NVIDIA-SMI 620.10.01    KMD Version: 620.10.01    CUDA UMD Version: 13.6     |"
COMPAT_CASES = [
    # name, nvidia-smi output, toolkit, exit code, printed
    ("compat: a 12.8 driver with the 12.9 toolkit needs the compat package", SMI_12_8, "12.9", 0, "compat 12.8"),
    ("compat: a 13.4 driver with the 13.4 toolkit runs natively", SMI_13_4, "13.4", 0, "native 13.4"),
    ("compat: a 13.6 driver with the 12.9 toolkit runs natively", SMI_13_6, "12.9", 0, "native 13.6"),
    ("compat: a 13.4 driver with the 12.9 toolkit runs natively", SMI_13_4, "12.9", 0, "native 13.4"),
    ("compat: a 12.8 driver with the 13.4 toolkit needs the compat package", SMI_12_8, "13.4", 0, "compat 12.8"),
    ("compat: nvidia-smi without a CUDA version is an error", "| NVIDIA-SMI 570.124.06 |", "12.9", 1, ""),
    ("compat: a toolkit that isn't major.minor is a usage error", SMI_13_4, "13", 2, ""),
]


def compat_case(name, smi, toolkit, code, printed):
    process = subprocess.run(["bash", CUDA_COMPAT, toolkit], input=f"Sat Oct 31 2026\n{smi}\n",
                             capture_output=True, text=True, timeout=60)
    actual = (process.returncode, process.stdout.strip())
    ok = actual == (code, printed)
    print(f"{'ok  ' if ok else 'FAIL'} {name}\n"
          f"     expected: exit {code}, {printed or 'nothing'}\n"
          f"     actual:   exit {actual[0]}, {actual[1] or 'nothing'}"
          + (f"\n     stderr: {process.stderr.strip()}" if process.stderr.strip() else ""))
    return ok


def show(values):
    action, reason, command, sha, backends, cuda_versions, accelerators, macos_windows = values
    return (f"{action} {reason} command={command or '-'} sha={sha or '-'} backends={backends or '-'} "
            f"cuda-versions={cuda_versions or '-'} accelerators={str(accelerators).lower()} "
            f"macos-windows={str(macos_windows).lower()}")


def main():
    results = []
    for name, ev, api_answers, changes, expected, expected_calls in CASES:
        api = test_requests.Fixtures(copy.deepcopy(api_answers))
        result = test_requests.decide(ev, {**SETTINGS, **changes}, api)
        actual = (result["action"], result["reason"], result["command"], result["sha"], result["backends"],
                  result["cuda_versions"], result["accelerators"], result["macos_windows"])
        ok = actual == expected
        if result["action"] == "accept":
            ok = ok and result["sha"] in result["reply"]
        if result["action"] == "accept" and result["command"] == "gpu" and api_answers["files"]:
            # Every accepted GPU reply names a changed file.
            ok = ok and any(f"`{file['filename']}`" in result["reply"]
                            or f"`{file.get('previous_filename')}`" in result["reply"]
                            for file in api_answers["files"])
        if result["reason"] == "no-gpu-changes":
            ok = ok and "`/gpu-tests all`" in result["reply"]
        if result["action"] == "ignore":
            ok = ok and not result["reply"]
        if expected_calls is not None:
            ok = ok and api.calls == expected_calls
        results.append(ok)
        print(f"{'ok  ' if ok else 'FAIL'} {name}\n"
              f"     expected: {show(expected)}\n"
              f"     actual:   {show(actual)}"
              + (f"\n     API calls: {api.calls or 'none'}" if expected_calls is not None else "")
              + (f"\n     reply: {result['reply']}" if result["reply"] else ""))
    results += [behind_case(*case) for case in BEHIND_CASES]
    results += [api_error_annotation(), permission_not_found()]
    results += [command_case(*case) for case in COMMAND_CASES]
    results += [compat_case(*case) for case in COMPAT_CASES]
    print(f"{sum(results)} of {len(results)} cases match")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
