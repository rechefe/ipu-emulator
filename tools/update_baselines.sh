#!/usr/bin/env bash
# Regenerate every non-timing baseline from the current tree, then list the
# baseline files that changed.
#
#   tools/update_baselines.sh                  # every baseline except the kernels'
#   tools/update_baselines.sh --with-kernels   # also every kernel's baseline.json (minutes)
#
# Run it only in a behaviour-change PR, and commit only the baselines that PR
# means to change, with one sentence of reason each (the rule is in
# docs/content/code-conventions.md, "Baselines"). A refactor PR changes no
# baseline file. The timing tests have no baseline file: their expected values
# are code, edited by hand. The *_WIDTH table in test_sv_package_golden.py is
# also edited by hand. //tools:test_update_baselines checks that UPDATE_TARGETS
# and KERNEL_TARGET below list every update_* target in the baseline packages.
set -euo pipefail

UPDATE_TARGETS=(
  //src/tools/ipu-isa:update_isa_baselines
  //src/tools/ipu-apps:update_asm_digests
  //src/tools/ipu-as-py:update_decode_snapshots
  //src/tools/ipu-apps:update_disassembly_snapshots
  //src/tools/ipu-as-py:update_sv_golden
  //src/tools/ipu-as-py:update_generated_artifacts
)
KERNEL_TARGET=//src/tools/ipu-apps:update_kernel_baselines

BASELINE_PATHS=(
  src/tools/ipu-isa/test/fixtures/spec_snapshot.json
  docs/content/requirements/faults.md
  src/tools/ipu-apps/test/baselines
  src/tools/ipu-as-py/test/baselines
  src/tools/ipu-as-py/test/golden
  ':(glob)src/tools/ipu-apps/src/ipu_apps/kernels/**/baseline.json'
)

usage() {
  echo "usage: $0 [--with-kernels]" >&2
  exit 2
}

with_kernels=0
for arg in "$@"; do
  case "$arg" in
    --with-kernels) with_kernels=1 ;;
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
if [[ $with_kernels -eq 1 ]]; then
  targets+=("$KERNEL_TARGET")
fi

# One build first, so each run below starts at once.
bazel build "${targets[@]}"
for target in "${targets[@]}"; do
  echo "== $target"
  bazel run "$target"
done

echo
echo "Baseline files changed (empty: none):"
git status --short -- "${BASELINE_PATHS[@]}"
