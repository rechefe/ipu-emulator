#!/usr/bin/env bash
# Regenerate every non-timing baseline from the current tree, then list the
# baseline files that changed.
#
#   tools/update_baselines.sh
#
# Run it only in a behaviour-change PR, and commit only the baselines that PR
# means to change, with one sentence of reason each (the rule is in
# docs/content/reference/code-conventions.md, "Baselines"). A refactor PR changes no
# baseline file. The timing tests have no baseline file: their expected values
# are code, edited by hand. The *_WIDTH table in test_sv_package_golden.py is
# also edited by hand. //tools:test_update_baselines checks that UPDATE_TARGETS
# below lists every update_* target in the baseline packages.
set -euo pipefail

UPDATE_TARGETS=(
  //src/ipu-isa:update_isa_baselines
  //src/ipu-as:update_asm_digests
  //src/ipu-as:update_decode_snapshots
  //src/ipu-as:update_disassembly_snapshots
  //src/ipu-gen:update_sv_golden
  //src/ipu-gen:update_generated_artifacts
  //src/ipu-emu:update_conformance
)

BASELINE_PATHS=(
  src/ipu-isa/test/baselines
  docs/content/requirements/faults.md
  src/ipu-as/test/baselines
  src/ipu-gen/test/baselines
  src/ipu-debug/test/baselines
  src/ipu-isa/conformance
)

usage() {
  echo "usage: $0" >&2
  exit 2
}

for arg in "$@"; do
  case "$arg" in
    -h | --help) usage ;;
    *)
      echo "unknown argument: $arg" >&2
      usage
      ;;
  esac
done

repo="$(git rev-parse --show-toplevel)"
cd "$repo"

targets=("${UPDATE_TARGETS[@]}")

# One build first, so each run below starts at once.
bazel build "${targets[@]}"
for target in "${targets[@]}"; do
  echo "== $target"
  bazel run "$target"
done

echo
echo "Baseline files changed (empty: none):"
git status --short -- "${BASELINE_PATHS[@]}"
