#!/usr/bin/env bash
# Checks with clang-format the lines of src/, test/ and examples/ that changed
# since a base commit.
#
# Usage: lint.sh [<base>]              check the working tree
#        lint.sh --diff [<base>]       print the changed lines as a diff, for --check
#        lint.sh --check <diff file>   check the lines a diff from --diff names, in a
#                                      copy of the tree that has no git history
#   <base> defaults to $AF_LINT_BASE. When it is empty, all zeros or not in the
#   local history (a new branch or a force push), the merge-base with the
#   default branch is used instead.
set -euo pipefail

clang_format_version=15.0.7
mode=check-tree
case ${1:-} in
  --diff) mode=print-diff; shift ;;
  --check) mode=check-diff; diff_file=$2; shift 2 ;;
esac

if [[ $mode != check-diff ]]; then
  cd "$(git rev-parse --show-toplevel)"

  is_commit() {
    [[ -n $1 && ! $1 =~ ^0+$ ]] && git rev-parse -q --verify "$1^{commit}" > /dev/null
  }

  base=${1:-${AF_LINT_BASE:-}}
  if ! is_commit "$base"; then
    [[ -n $base ]] && echo "Base $base is not available locally." >&2
    base=
    for ref in origin/HEAD origin/master master; do
      if is_commit "$ref"; then
        base=$(git merge-base HEAD "$ref")
        echo "Using the merge-base with $ref." >&2
        break
      fi
    done
    if [[ -z $base ]]; then
      echo "error: no base commit to compare against" >&2
      exit 2
    fi
  fi
fi

if [[ $mode == print-diff ]]; then
  # Read only: no index refresh, no external diff or textconv programs.
  echo "Lines changed since $(git rev-parse --short "$base")" >&2
  exec git --no-optional-locks -c diff.autoRefreshIndex=false diff -U0 --no-color \
    --no-ext-diff --no-textconv "$base" -- src test examples
fi

# The PyPI wheel gives the same clang-format build on every platform.
venv=${XDG_CACHE_HOME:-$HOME/.cache}/arrayfire/clang-format-$clang_format_version
bin=$venv/bin
[[ -d $venv/Scripts ]] && bin=$venv/Scripts
if [[ ! -x $bin/git-clang-format ]]; then
  python3 -m venv "$venv"
  [[ -d $venv/Scripts ]] && bin=$venv/Scripts
  "$bin/python" -m pip install --quiet --disable-pip-version-check \
    "clang-format==$clang_format_version"
fi

if [[ $mode == check-diff ]]; then
  echo "Checking the lines changed in $diff_file with $("$bin/clang-format" --version)"
  formatted=$("$bin/python" "$bin/clang-format-diff.py" -p1 -binary "$bin/clang-format" \
    -iregex '.*\.(h|hpp|cpp)' < "$diff_file")
  if [[ -n $formatted ]]; then
    echo "$formatted"
    echo "Formatting differs: the diff above shows the changes clang-format wants."
    exit 1
  fi
  exit 0
fi

echo "Checking lines changed since $(git rev-parse --short "$base") with $("$bin/clang-format" --version)"
rc=0
"$bin/git-clang-format" --binary "$bin/clang-format" --extensions h,hpp,cpp \
  --diff "$base" -- src test examples || rc=$?
if [[ $rc -eq 1 ]]; then
  echo "Formatting differs. To fix it locally, run:"
  echo "  $bin/git-clang-format --binary $bin/clang-format $(git rev-parse --short "$base")"
fi
exit $rc
