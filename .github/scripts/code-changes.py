#!/usr/bin/env python3
"""Decides whether a cpu.yml run needs the CPU tests.

  code-changes.py   decide on the event in GITHUB_EVENT_PATH and write code (true or false)
                    and reason to GITHUB_OUTPUT, and the reason to the log and step summary

Only a merge-queue entry or a push to master that adds or changes nothing but non-code files
(NON_CODE) skips the CPU tests. Everything else runs them in full: a requested commit (SHA,
from /cpu-tests or a manual run), a manual run of a branch, a removed or renamed file, any
other file, and a comparison that GitHub can't make or list in full.

A merge-queue entry is compared with the branch it merges into, not with the group's parent
commit (base_sha): behind another entry, that parent is the entry ahead, whose changes merge
together with this one's.

With AF_CI_CHANGES_FIXTURES naming a JSON file of the comparison GitHub would return, it
replaces GitHub; that is honoured only in act runs (ACT=true).
"""
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request

# The files that no build, test, installer or release reads; every other file is code,
# including the rest of .github/. Markdown at the top level; docs/, which only the docs
# check builds, and it runs on every change; GitHub's issue and pull-request templates and
# citation files; git's .gitignore and .mailmap at the top level. Only additions and
# changes count: CPack checks at every configure on Linux that README.md exists.
NON_CODE = re.compile(r"[^/]+\.md|docs/.+|\.github/ISSUE_TEMPLATE/.+|\.github/pull_request_template\.md"
                      r"|\.github/CITATION\.md|CITATION\.cff|\.gitignore|\.mailmap")
KEPT = {"added", "modified", "changed", "copied"}  # statuses that leave every old path in place
LISTED = 300  # GitHub's comparison lists at most this many files
NAMED = 10  # files a reason names
FULL = "the CPU tests run in full."
ZERO = "0" * 40


def named(paths):
    """The paths, at most NAMED of them, as "a, b and 3 more". Line breaks in a name are
    escaped, so that no name starts a line of the log, as a workflow command would."""
    shown = ", ".join(path.replace("\r", "\\r").replace("\n", "\\n") for path in paths[:NAMED])
    return shown + (f" and {len(paths) - NAMED} more" if len(paths) > NAMED else "")


def classify(files):
    """(code, reason) for a comparison's files: [{"filename", "status", "previous_filename"}]."""
    if not files:
        return True, f"No changed files were listed: {FULL}"
    code = [file["filename"] if file.get("status") in KEPT else f"{file['filename']} {file.get('status') or 'unlisted'}"
            for file in files if file.get("status") not in KEPT or not NON_CODE.fullmatch(file["filename"])]
    if code:
        return True, f"Code changed ({named(code)}): {FULL}"
    return False, (f"Only non-code files changed ({named([file['filename'] for file in files])}): the "
                   "CPU tests are skipped. Changes to any other file run them in full.")


def decide(event_name, event, sha, api):
    """(code, reason) for one run of cpu.yml; asks api only for a merge-queue entry or a push."""
    if sha:
        return True, f"Commit {sha} was requested: {FULL}"
    if event_name == "merge_group":
        group = event.get("merge_group") or {}
        base = (group.get("base_ref") or "").removeprefix("refs/heads/")
        head = group.get("head_sha") or ""
        what = f"merge group {head[:9]} and {base}"
    elif event_name == "push":
        base, head = event.get("before") or "", event.get("after") or ""
        if base in ("", ZERO):
            return True, f"The push has no earlier commit to compare with: {FULL}"
        what = f"{base[:9]} and {head[:9]}"
    else:
        return True, f"A {event_name} run was requested: {FULL}"
    if not base or not head:
        return True, f"The event names no commits to compare: {FULL}"
    try:
        comparison = api.compare(base, head)
        status, files = comparison["status"], comparison.get("files") or []
    except (OSError, KeyError, TypeError, ValueError) as error:
        annotate("warning", f"Can't compare {what}: {error}")
        return True, f"GitHub couldn't compare {what}: {FULL}"
    if status != "ahead":
        return True, f"The comparison of {what} is {status}, not ahead: {FULL}"
    if len(files) >= LISTED:
        return True, f"GitHub lists {len(files)} changed files, which may not be all of them: {FULL}"
    return classify(files)


def annotate(level, text):
    """Prints text as an annotation of the run (level is error or warning)."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level}::{text}", flush=True)


class GitHub:
    """The one REST call: the comparison of two commits, with the run's read-only token."""

    def __init__(self, repo, token):
        self.repo = repo
        self.token = token

    def compare(self, base, head):
        request = urllib.request.Request(
            f"https://api.github.com/repos/{self.repo}/compare/{urllib.parse.quote(base, safe='')}...{head}",
            headers={"Accept": "application/vnd.github+json",
                     "Authorization": f"Bearer {self.token}",
                     "X-GitHub-Api-Version": "2022-11-28"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise OSError(f"HTTP {error.code}: {error.reason}") from None


class Fixtures:
    """The comparison from a JSON file, for act runs."""

    def __init__(self, comparison):
        self.comparison = comparison

    def compare(self, base, head):
        print(f"[fixture] compare {base}...{head}")
        return self.comparison


def api_from_env():
    fixtures = os.environ.get("AF_CI_CHANGES_FIXTURES")
    if fixtures:
        if os.environ.get("ACT") != "true":
            sys.exit("AF_CI_CHANGES_FIXTURES is honoured only in act runs (ACT=true)")
        with open(fixtures) as file:
            return Fixtures(json.load(file))
    return GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])


def main():
    with open(os.environ["GITHUB_EVENT_PATH"]) as file:
        event = json.load(file)
    code, reason = decide(os.environ.get("GITHUB_EVENT_NAME", ""), event, os.environ.get("SHA", ""),
                          api_from_env())
    print(f"code={str(code).lower()}: {reason}")
    delimiter = f"END_{secrets.token_hex(16)}"
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"code={str(code).lower()}\nreason<<{delimiter}\n{reason}\n{delimiter}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(reason + "\n")


if __name__ == "__main__":
    main()
