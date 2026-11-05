#!/usr/bin/env bash
# Counts the archives in vcpkg's binary cache directory, so that CI saves the cache
# only when a run added archives. Prints the count and, when GITHUB_OUTPUT is set,
# writes count=<n> to it. Given the count from before the build, it also writes
# new=true if there are more archives now, else new=false.
#
# Usage: count-vcpkg-archives.sh <archive dir> [<count before the build>]
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || { echo "usage: $0 <archive dir> [<count before the build>]" >&2; exit 2; }
dir=$1
before=${2:-}

count=0
if [[ -d $dir ]]; then
  count=$(find "$dir" -type f -name '*.zip' | wc -l)
  count=$((count))
fi

if [[ -z $before ]]; then
  echo "vcpkg archives in $dir: $count"
else
  new=false
  ((count > before)) && new=true
  echo "vcpkg archives in $dir: $count, $((count - before)) more than before the build"
fi
if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  echo "count=$count" >> "$GITHUB_OUTPUT"
  [[ -z $before ]] || echo "new=$new" >> "$GITHUB_OUTPUT"
fi
