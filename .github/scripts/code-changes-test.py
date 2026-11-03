#!/usr/bin/env python3
"""Tests of code-changes.py: which changed files skip the CPU tests, which events may skip
them at all, and what the script writes for cpu.yml.
"""
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
SCRIPT = pathlib.Path(__file__).resolve().parent / "code-changes.py"
spec = importlib.util.spec_from_file_location("code_changes", SCRIPT)
changes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(changes)

BEFORE = "ab4a5063c92512dd65ab2b418cf81e4b3c46299b"
AFTER = "d2f542b6d6376dd548e470ed7e367a66f78597dc"
GROUP = "8c68aa7d4c0e6f1a2b3c4d5e6f708192a3b4c5d6"
PARENT = "411bdd0dc2cff65f9faa873db7232c52b819799b"


def modified(*paths):
    return [{"filename": path, "status": "modified"} for path in paths]


class Api:
    """Answers compare with comparison, or raises error; records the calls."""

    def __init__(self, comparison=None, error=None):
        self.comparison = comparison
        self.error = error
        self.calls = []

    def compare(self, base, head):
        self.calls.append(f"{base}...{head}")
        if self.error:
            raise self.error
        return self.comparison


def ahead(files):
    return {"status": "ahead", "files": files}


def push(before=BEFORE):
    return {"before": before, "after": AFTER}


def merge_group():
    return {"merge_group": {"base_ref": "refs/heads/master", "base_sha": PARENT, "head_sha": GROUP,
                            "head_ref": "refs/heads/gh-readonly-queue/master/pr-3698-" + PARENT}}


def classified(files, code, *words):
    """A case: classify(files) gives code, with every word in the reason."""
    def case():
        got, reason = changes.classify(files)
        return got == code and all(word in reason for word in words), f"code={str(got).lower()}: {reason}"
    return case


def decided(event_name, event, sha, api, code, calls, *words):
    """A case: decide() gives code, with every word in the reason, after exactly these calls."""
    def case():
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            got, reason = changes.decide(event_name, event, sha, api)
        ok = got == code and api.calls == calls and all(word in reason + printed.getvalue() for word in words)
        return ok, (f"code={str(got).lower()}: {reason}\nAPI calls: {api.calls}"
                    + (f"\nprinted: {printed.getvalue().strip()}" if printed.getvalue() else ""))
    return case


def run_script(event_name, event, comparison, act="true", sha=""):
    """Runs the script as cpu.yml does; returns its exit code, output, GITHUB_OUTPUT and summary."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "event.json").write_text(json.dumps(event))
        (tmp / "comparison.json").write_text(json.dumps(comparison))
        env = {"PATH": os.environ.get("PATH", ""), "GITHUB_EVENT_NAME": event_name,
               "GITHUB_EVENT_PATH": str(tmp / "event.json"), "GITHUB_OUTPUT": str(tmp / "output"),
               "GITHUB_STEP_SUMMARY": str(tmp / "summary"), "SHA": sha, "ACT": act,
               "AF_CI_CHANGES_FIXTURES": str(tmp / "comparison.json")}
        process = subprocess.run([sys.executable, "-I", str(SCRIPT)], env=env, capture_output=True,
                                 text=True, timeout=60)
        read = lambda name: (tmp / name).read_text() if (tmp / name).exists() else ""
        return process.returncode, process.stdout + process.stderr, read("output"), read("summary")


def case_script_skips():
    rc, out, output, summary = run_script("merge_group", merge_group(), ahead(modified("README.md")))
    ok = (rc == 0 and output.startswith("code=false\nreason<<END_") and "Only non-code files changed (README.md)"
          in output and summary.startswith("Only non-code files changed (README.md): the CPU tests are skipped."))
    return ok, f"rc={rc}\n{out.strip()}\nGITHUB_OUTPUT: {output.strip()}\nsummary: {summary.strip()}"


def case_script_runs():
    rc, out, output, summary = run_script("push", push(), ahead(modified("src/api/c/abs.cpp")))
    ok = rc == 0 and output.startswith("code=true\n") and summary.startswith("Code changed (src/api/c/abs.cpp")
    return ok, f"rc={rc}\n{out.strip()}\nGITHUB_OUTPUT: {output.strip()}"


def case_fixtures_need_act():
    rc, out, output, _ = run_script("push", push(), ahead(modified("README.md")), act="")
    ok = rc != 0 and "only in act runs" in out and not output
    return ok, f"rc={rc}: {out.strip()}"


MANY = modified(*[f"docs/pages/page{n}.md" for n in range(12)])

CASES = [
    # Which files are non-code.
    ("only README.md: skip", classified(modified("README.md"), False,
     "Only non-code files changed (README.md): the CPU tests are skipped. Changes to any other file run them in full.")),
    ("README.md and a source file: run", classified(modified("README.md", "src/api/c/abs.cpp"), True,
     "Code changed (src/api/c/abs.cpp): the CPU tests run in full.", "run in full")),
    ("only docs/: skip", classified(modified("docs/pages/install.md", "docs/details/algorithm.dox",
                                             "docs/doxygen.mk"), False, "docs/doxygen.mk")),
    ("LICENSE: run (CMake configures it on macOS and Windows; CPack checks it on Linux)",
     classified(modified("LICENSE"), True, "Code changed (LICENSE)")),
    ("LICENSES/: run (installed)", classified(modified("LICENSES/BSD 3-Clause.txt"), True, "LICENSES/")),
    (".github/workflows/cpu.yml: run", classified(modified(".github/workflows/cpu.yml"), True,
                                                  ".github/workflows/cpu.yml")),
    ("an issue template: skip", classified(modified(".github/ISSUE_TEMPLATE/bug_report.md"), False,
                                           ".github/ISSUE_TEMPLATE/bug_report.md")),
    ("the PR template, citation files, .gitignore and .mailmap: skip",
     classified(modified(".github/pull_request_template.md", ".github/CITATION.md", "CITATION.cff",
                         ".gitignore", ".mailmap", "CONTRIBUTING.md"), False, ".mailmap")),
    ("Markdown below the top level: run (examples/ is installed)",
     classified(modified("examples/README.md", "src/backend/cuda/README.md"), True, "examples/README.md")),
    ("CI's own Markdown and nested .gitignore files: run",
     classified(modified(".github/fixtures/README.md", "test/.gitignore"), True, ".github/fixtures/README.md")),
    ("an empty list: run, with its own reason", classified([], True, "No changed files were listed")),
    ("README.md removed: run (CPack checks that it exists)",
     classified([{"filename": "README.md", "status": "removed"}], True, "Code changed (README.md removed)")),
    ("a renamed doc: run", classified([{"filename": "docs/pages/new.md", "status": "renamed",
                                        "previous_filename": "docs/pages/old.md"}], True, "docs/pages/new.md renamed")),
    ("a file without a status: run", classified([{"filename": "README.md"}], True, "README.md unlisted")),
    ("a skip names 10 files and counts the rest", classified(MANY, False, "page9.md and 2 more")),
    ("a name with a line break: run, and it is escaped", classified(modified("docs/a\n::error::b.md"), True,
                                                                    "(docs/a\\n::error::b.md)")),
    # Which events may skip, and what they compare.
    ("push, zero before SHA: run, without asking GitHub",
     decided("push", push("0" * 40), "", Api(), True, [], "no earlier commit")),
    ("push, docs only: skip, comparing before with after",
     decided("push", push(), "", Api(ahead(modified("docs/pages/jit.md"))), False,
             [f"{BEFORE}...{AFTER}"], "docs/pages/jit.md")),
    ("push, code: run", decided("push", push(), "", Api(ahead(modified("CMakeLists.txt"))), True,
                                [f"{BEFORE}...{AFTER}"], "CMakeLists.txt")),
    ("merge group, docs only: skip, comparing the group with master, not with base_sha",
     decided("merge_group", merge_group(), "", Api(ahead(modified("README.md"))), False,
             [f"master...{GROUP}"], "README.md")),
    ("merge group, code: run", decided("merge_group", merge_group(), "", Api(ahead(modified("src/api/c/abs.cpp"))),
                                       True, [f"master...{GROUP}"], "src/api/c/abs.cpp")),
    ("/cpu-tests (workflow_call from an issue comment, with a SHA): run, without asking GitHub",
     decided("issue_comment", {}, AFTER, Api(ahead(modified("README.md"))), True, [], f"Commit {AFTER}")),
    ("workflow_dispatch with a SHA: run", decided("workflow_dispatch", {}, AFTER, Api(), True, [], "requested")),
    ("workflow_dispatch of a branch: run", decided("workflow_dispatch", {}, "", Api(), True, [],
                                                   "A workflow_dispatch run was requested")),
    ("a push with a SHA input can't skip either", decided("push", push(), AFTER, Api(ahead(modified("README.md"))),
                                                         True, [], "requested")),
    ("a failed comparison: run, with a warning",
     decided("push", push(), "", Api(error=OSError("HTTP 404: Not Found")), True, [f"{BEFORE}...{AFTER}"],
             "::warning::Can't compare", "couldn't compare")),
    ("a diverged push: run", decided("push", push(), "", Api({"status": "diverged", "files": modified("README.md")}),
                                     True, [f"{BEFORE}...{AFTER}"], "is diverged")),
    ("300 listed files: run", decided("push", push(), "", Api(ahead(modified(*[f"docs/{n}.md" for n in range(300)]))),
                                      True, [f"{BEFORE}...{AFTER}"], "may not be all")),
    ("a merge group without a head SHA: run",
     decided("merge_group", {"merge_group": {"base_ref": "refs/heads/master"}}, "", Api(), True, [], "names no commits")),
    # The script as cpu.yml runs it.
    ("the script writes code=false, the reason and the summary", case_script_skips),
    ("the script writes code=true for a code change", case_script_runs),
    ("fixtures outside act are refused", case_fixtures_need_act),
]


def main():
    results = []
    for name, case in CASES:
        ok, detail = case()
        results.append(ok)
        print(f"{'ok  ' if ok else 'FAIL'} {name}\n     " + detail.replace("\n", "\n     "))
    print(f"{sum(results)} of {len(results)} cases match")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
