#!/usr/bin/env python3
"""Tests of request-tests.sh with a stub gh, which records its arguments instead of calling
GitHub.

Each case is the script's arguments and the event it runs on; the test prints the expected
and the actual exit code and gh call, and fails if any case differs.
"""
import json
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "request-tests.sh")
REPO = "arrayfire/arrayfire"
TOKEN = "test-token-not-a-credential"
STUB = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["GH_STUB_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
print("https://github.com/arrayfire/arrayfire/pull/0#issuecomment-1")
"""


def pr_comment(number):
    return {"action": "created", "issue": {"number": number, "pull_request": {"url": "..."}},
            "comment": {"body": "@claude request the CPU tests"}}


def review_comment(number):
    return {"action": "created", "pull_request": {"number": number},
            "comment": {"body": "@claude request the CPU tests"}}


def issue_comment(number):
    return {"action": "created", "issue": {"number": number}, "comment": {"body": "@claude"}}


def issue_opened(number):
    return {"action": "opened", "issue": {"number": number, "body": "@claude"}}


def posted(number, body):
    return ["api", "--method", "POST", f"repos/{REPO}/issues/{number}/comments", "-f", f"body={body}",
            "--jq", ".html_url"]


USAGE = "usage: "
NO_PR = "no pull request: open one first"
CASES = [
    # name, arguments, event (None: no GITHUB_EVENT_PATH), exit code, gh call (None: no call),
    # text that must be in the output
    ("cpu on a pull request comment", ["cpu"], pr_comment(3), 0, posted(3, "/cpu-tests"),
     "Requested /cpu-tests on pull request #3"),
    ("cpu all", ["cpu", "all"], pr_comment(3), 0, posted(3, "/cpu-tests all"), "#3"),
    ("gpu", ["gpu"], pr_comment(3), 0, posted(3, "/gpu-tests"), "#3"),
    ("gpu cuda=13.1", ["gpu", "cuda=13.1"], pr_comment(3), 0, posted(3, "/gpu-tests cuda=13.1"), "#3"),
    ("a review comment's pull request", ["cpu"], review_comment(7), 0, posted(7, "/cpu-tests"), "#7"),
    ("a comment on an issue, not a pull request", ["cpu"], issue_comment(5), 1, None, NO_PR),
    ("an opened issue", ["gpu"], issue_opened(5), 1, None, NO_PR),
    ("no event", ["cpu"], None, 1, None, "no GitHub event"),
    ("no arguments", [], pr_comment(3), 2, None, USAGE),
    ("an extra argument", ["cpu", "all", "now"], pr_comment(3), 2, None, USAGE),
    ("an unknown second argument", ["cpu", "everything"], pr_comment(3), 2, None, USAGE),
    ("an unknown command", ["tests"], pr_comment(3), 2, None, USAGE),
    ("uppercase", ["CPU"], pr_comment(3), 2, None, USAGE),
    ("'cpu all' as one argument", ["cpu all"], pr_comment(3), 2, None, USAGE),
    ("'cpu; echo x' as one argument", ["cpu; echo x"], pr_comment(3), 2, None, USAGE),
    ("a command after the request", ["cpu", "all", ";", "echo", "x"], pr_comment(3), 2, None, USAGE),
    ("cuda without a minor version", ["gpu", "cuda=13"], pr_comment(3), 2, None, USAGE),
    ("cuda with a command", ["gpu", "cuda=13.1; cat /proc/self/environ"], pr_comment(3), 2, None, USAGE),
    ("cuda for the CPU tests", ["cpu", "cuda=13.1"], pr_comment(3), 2, None, USAGE),
    ("all for the GPU tests", ["gpu", "all"], pr_comment(3), 2, None, USAGE),
]


def run(arguments, event):
    with tempfile.TemporaryDirectory() as tmp:
        stub = os.path.join(tmp, "bin", "gh")
        os.makedirs(os.path.dirname(stub))
        with open(stub, "w") as file:
            file.write(STUB)
        os.chmod(stub, 0o755)
        env = {"PATH": f"{os.path.dirname(stub)}:{os.environ.get('PATH', '')}",
               "GH_STUB_LOG": os.path.join(tmp, "gh.log"), "GH_TOKEN": TOKEN, "GITHUB_REPOSITORY": REPO}
        if event is not None:
            env["GITHUB_EVENT_PATH"] = os.path.join(tmp, "event.json")
            with open(env["GITHUB_EVENT_PATH"], "w") as file:
                json.dump(event, file)
        process = subprocess.run(["bash", SCRIPT, *arguments], capture_output=True, text=True,
                                 timeout=60, env=env)
        calls = []
        if os.path.exists(env["GH_STUB_LOG"]):
            calls = [json.loads(line) for line in open(env["GH_STUB_LOG"])]
    return process.returncode, calls, process.stdout + process.stderr


def main():
    failed = 0
    for name, arguments, event, code, call, text in CASES:
        actual_code, calls, output = run(arguments, event)
        expected_calls = [call] if call else []
        ok = (actual_code == code and calls == expected_calls and text in output
              and TOKEN not in output and "x" not in output.splitlines())
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: request-tests.sh {' '.join(repr(a) for a in arguments)}\n"
              f"     expected: exit {code}, gh {expected_calls or 'not called'}\n"
              f"     actual:   exit {actual_code}, gh {calls or 'not called'}\n"
              f"     output: {output.strip().splitlines()[0] if output.strip() else ''}")
    print(f"{len(CASES) - failed} of {len(CASES)} cases match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
