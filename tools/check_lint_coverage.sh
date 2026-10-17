#!/usr/bin/env bash
# Every Python file in git is checked by exactly one ruff_test (tools/lint.bzl),
# and //tools:lint lists every ruff_test. Run from CI outside the Bazel sandbox
# (it needs git):
#
#   tools/check_lint_coverage.sh
#
# A new package needs a `ruff_test(name = "ruff", srcs = glob(["**/*.py"]))`
# in its BUILD.bazel and an entry in //tools:lint.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Tracked .py files that exist in the working tree (a deleted, uncommitted
# file has no package to cover it).
git ls-files -z '*.py' | while IFS= read -r -d '' f; do
  [ -f "$f" ] && printf '%s\n' "$f"
done | LC_ALL=C sort >"$tmp/git"

# Files in the srcs of every ruff_test (also the ones ipu_kernel() declares,
# whose generator_function is the outer macro), as workspace-relative paths:
# //pkg:sub/file.py -> pkg/sub/file.py, //:file.py -> file.py.
bazel query --noshow_progress \
  'labels(srcs, kind(filegroup, attr(generator_function, "^ruff_test$", //...) + attr(name, "^ruff_files$", attr(generator_function, "^ipu_kernel$", //kernels/...))))' \
  | sed -e 's#^//:##' -e 's#^//\([^:]*\):#\1/#' \
  | grep '\.py$' | LC_ALL=C sort >"$tmp/covered" || true

status=0

missing="$(LC_ALL=C comm -23 "$tmp/git" <(LC_ALL=C sort -u "$tmp/covered"))"
if [ -n "$missing" ]; then
  echo "error: Python files not covered by any ruff_test (add one to the package's BUILD.bazel):" >&2
  sed 's/^/  /' <<<"$missing" >&2
  status=1
fi

duplicated="$(uniq -d "$tmp/covered")"
if [ -n "$duplicated" ]; then
  echo "error: Python files covered by more than one ruff_test:" >&2
  sed 's/^/  /' <<<"$duplicated" >&2
  status=1
fi

bazel query --noshow_progress \
  'kind(py_test, attr(generator_function, "^ruff_test$", //...) + attr(name, "^ruff$", attr(generator_function, "^ipu_kernel$", //kernels/...)))' \
  | LC_ALL=C sort >"$tmp/targets"
bazel query --noshow_progress 'tests(//tools:lint)' | LC_ALL=C sort >"$tmp/suite"
if ! diff -u --label "ruff_test targets" --label "tests(//tools:lint)" \
  "$tmp/targets" "$tmp/suite" >"$tmp/suite.diff"; then
  echo "error: //tools:lint (tools/BUILD.bazel) does not list exactly the ruff_test targets:" >&2
  cat "$tmp/suite.diff" >&2
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "ruff_test coverage: $(wc -l <"$tmp/git") Python files in $(wc -l <"$tmp/targets") targets"
fi
exit "$status"
