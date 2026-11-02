#!/usr/bin/env bash
# Requests the CPU or GPU tests of the pull request that this run is about, by
# commenting "/cpu-tests" or "/gpu-tests" on it (test-requests.yml). claude.yml lets
# Claude comment only through this script, so a comment from it can only be one of
# these requests. The pull request comes from the event in GITHUB_EVENT_PATH; gh
# authenticates with GH_TOKEN, which claude-code-action sets to its token.
#
# Usage: request-tests.sh cpu [all]
#        request-tests.sh gpu [cuda=<major.minor>]
set -euo pipefail

usage() {
  echo "usage: $0 cpu [all]" >&2
  echo "       $0 gpu [cuda=<major.minor>]" >&2
  exit 2
}

if [[ $# -eq 1 && $1 == cpu ]]; then
  body=/cpu-tests
elif [[ $# -eq 2 && $1 == cpu && $2 == all ]]; then
  body="/cpu-tests all"
elif [[ $# -eq 1 && $1 == gpu ]]; then
  body=/gpu-tests
elif [[ $# -eq 2 && $1 == gpu && $2 =~ ^cuda=[0-9]+\.[0-9]+$ ]]; then
  body="/gpu-tests $2"
else
  usage
fi

[[ -f ${GITHUB_EVENT_PATH:-} ]] || { echo "no GitHub event to find the pull request in (GITHUB_EVENT_PATH)" >&2; exit 1; }
number=$(python3 -c '
import json, sys
event = json.load(open(sys.argv[1]))
issue = event.get("issue") or {}
pull = issue if issue.get("pull_request") else event.get("pull_request") or {}
print(pull.get("number") or "")' "$GITHUB_EVENT_PATH")
[[ $number =~ ^[0-9]+$ ]] || { echo "no pull request: open one first" >&2; exit 1; }

url=$(gh api --method POST "repos/${GITHUB_REPOSITORY:?}/issues/$number/comments" -f body="$body" --jq .html_url)
echo "Requested $body on pull request #$number: $url"
