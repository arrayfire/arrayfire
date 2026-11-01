#!/usr/bin/env bash
# Checks that the scripts the macOS job runs work with macOS's bash 3.2: it runs
# itself in the official bash:3.2 image and checks their syntax, ci.sh --list,
# checkout-commit.sh's handling of a short SHA, and that they use no syntax added
# in bash 4. It must itself run on bash 3.2.
#
# The image is pinned by digest. To update it: docker pull bash:3.2, then
# docker inspect --format '{{index .RepoDigests 0}}' bash:3.2.
#
# Usage: check-bash32.sh [<repository>]   (default: this one)
set -euo pipefail

image=bash:3.2@sha256:b6252c62fc86114f93f02104a486c71e6569371d95950ea7479a2ffd6b47f16a
scripts="checkout-commit.sh ci.sh install-deps-macos.sh"

if [[ ${1:-} != --inside ]]; then
  root=$(cd "${1:-$(dirname "$0")/../..}" && pwd)
  exec docker run --rm --network none --memory=256m --memory-swap=256m \
    -v "$root:/ci:ro" "$image" bash /ci/.github/scripts/check-bash32.sh --inside
fi

cd /ci/.github/scripts
echo "bash $BASH_VERSION"
failed=0
fail() { echo "FAIL: $*"; failed=1; }

for script in $scripts; do
  bash -n "$script" || fail "bash -n $script"
done
./ci.sh --list > /dev/null || fail "ci.sh --list"
out=$(./checkout-commit.sh 0123ABC /tmp/checkout 2>&1) || true
[[ $out == "not a full 40-character commit SHA: 0123ABC" ]] ||
  fail "checkout-commit.sh with a short SHA printed: $out"

# Case conversion, associative arrays and namerefs, mapfile, -v tests, @
# transformations, &>> and |&, negative indices, wait -n, coproc, ;& and ;;&.
bash4='\$\{[A-Za-z_0-9]+(,,?|\^\^?)\}|\$\{[A-Za-z_0-9]+@[QEPAKa]\}|(declare|local|typeset) -[a-zA-Z]*[Anlu]|mapfile|readarray|\[\[ -v |test -v |&>>|\|&|\[-[0-9]+\]|wait -n|coproc|;;?&'
# shellcheck disable=SC2086
if grep -q -E "$bash4" $scripts; then
  fail "syntax that needs bash 4:"
  grep -n -E "$bash4" $scripts | sed 's/^/  /'
fi

if ((failed)); then
  exit 1
fi
echo "The macOS job's scripts ($scripts) pass under bash 3.2."
