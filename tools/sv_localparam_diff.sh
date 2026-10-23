#!/usr/bin/env bash
# Diff the SV package's localparam lines between <base-ref> and the working tree.
#
#   tools/sv_localparam_diff.sh <base-ref>
#
# Builds //docs:generate_instruction_format_artifacts in both trees (the base
# in a git worktree at a fixed path, so its Bazel output base is reused on the
# next run) and diffs the `localparam` lines of ipu_instr_pkg.sv. Prints
# nothing and exits 0 when they are identical; prints the diff and exits 1
# otherwise; exits 2 on a usage or build error. Every *_WIDTH is wire format
# for the downstream RTL repo (CLAUDE.md, "Generated SystemVerilog Package").
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <base-ref>" >&2
  exit 2
fi

repo="$(git rev-parse --show-toplevel)" || exit 2
base="$(git -C "$repo" rev-parse --verify "$1^{commit}")" || exit 2
worktree="$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)/sv-localparam-diff"
target="//docs:generate_instruction_format_artifacts"
scratch="$(mktemp -d)"

cleanup() {
  git -C "$repo" worktree remove --force "$worktree" >/dev/null 2>&1 || true
  rm -rf "$scratch"
}
trap cleanup EXIT

# Build $target in $1 and write its localparam lines to $2.
localparams() {
  local tree="$1" out="$2" log="$scratch/build.log"
  if ! (cd "$tree" && bazel build --noshow_progress "$target") >"$log" 2>&1; then
    cat "$log" >&2
    echo "$0: building $target in $tree failed" >&2
    exit 2
  fi
  grep -E '^[[:space:]]*localparam[[:space:]]' "$tree/bazel-bin/docs/ipu_instr_pkg.sv" | sed -E 's/^[[:space:]]+//' >"$out"
}

git -C "$repo" worktree remove --force "$worktree" >/dev/null 2>&1 || true
if ! git -C "$repo" worktree add --force --detach "$worktree" "$base" >"$scratch/worktree.log" 2>&1; then
  cat "$scratch/worktree.log" >&2
  echo "$0: git worktree add failed" >&2
  exit 2
fi

localparams "$worktree" "$scratch/base.txt"
localparams "$repo" "$scratch/head.txt"

diff -u --label "localparams @ $1" --label "localparams @ working tree" \
  "$scratch/base.txt" "$scratch/head.txt" || exit 1
