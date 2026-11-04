#!/usr/bin/env bash
# Checks out one commit of the GitHub repository into its own directory, apart
# from the checkout that holds the CI scripts. The fetch is anonymous and
# shallow and keeps no credentials. It must run on macOS's bash 3.2.
#
# Usage: checkout-commit.sh <40-character commit SHA> <dir>
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 <sha> <dir>" >&2; exit 2; }
sha=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
dir=$2
[[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "not a full 40-character commit SHA: $1" >&2; exit 2; }
url=${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-arrayfire/arrayfire}

git init -q "$dir"
git -C "$dir" fetch -q --depth 1 --no-tags "$url" "$sha"
git -C "$dir" checkout -q --detach FETCH_HEAD
git -C "$dir" --no-pager log -1 --format='Checked out %H (%s)'
