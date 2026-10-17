# Retire `src/tools/ipu-apps` and its Bazel macros

Part of [Epic: Kernel migration](00-epic.md) · ID E07-25 · Size S · Depends on: E07-02..E07-24, E07-26, E06-20 · Behaviour change: none

## Goal

Delete the old harness package once nothing uses it. After this issue,
`ipu_harness` + `ipu_kernels` + `kernels/` are the only kernel framework.

## Background

What remains after the family migrations:
- `kernel_registry/`: 2,145 lines. That is `base.py`, `cases.py`,
  `testing.py`, `registry.py`, `discovery.py`, `layers.py`, `runner.py`,
  `bazel_entry.py`, `benchmarking.py`, `case_support.py`, `coverage.py`,
  `manifest.py`, `query.py`, `shapes.py` and `spec.py`, plus `memory.py`, which
  E06-20 removes.
- `ipu_app.bzl`: the `ipu_app`, `ipu_apps_from_kernels`,
  `ipu_benchmarks_from_kernels`, `ipu_families_from_kernels`,
  `ipu_multi_kernel_tests`, `ipu_per_kernel_test` and `ipu_kernel_targets`
  macros.
- `src/tools/ipu-apps/BUILD.bazel`, `pyproject.toml`,
  `test/test_kernel_registry.py`, `test/test_harness_factory.py`,
  `test/test_alias_profile_kernels.py`, `test/test_debug_launch.py` and
  `test/test_bazel_entry.py`.
- Docs: `docs/content/adding-applications.md`, `building-applications.md` and
  `app-coverage.md` refer to `//src/tools/ipu-apps:*` labels.

## Implementation

- Confirm with a grep that nothing imports `ipu_apps` or loads `//:ipu_app.bzl`.
- Delete the package, the macros, and the `ipu-apps` entry in the dependency
  generation (`MODULE.bazel` `generate_requirements` sources).
- Remove the temporary migration tools (E07-01 `compare_setup` and
  `compare_legacy`). Keep `upper_mnemonics.py --check` as a test over
  `kernels/`.
- Docs: replace the old pages with links to the E06-21 pages. E09 owns the
  final structure. Update `SKILL.md` and `CLAUDE.md` (`bazel test //src/tools/ipu-apps:test_fully_connected`
  becomes `bazel test //kernels/linear/fully_connected`).
- Any registry-level test that is still useful (coverage report, manifest)
  should already exist under `//kernels:*` by now. List any that were dropped,
  with reasons.

## Tests

- [ ] `bazel test //...` and `bazel test --config=full //...` (nightly tier) are green.
- [ ] `upper_mnemonics.py --check` over `kernels/` passes.
- [ ] A grep finds no `ipu_apps` and no `//src/tools/ipu-apps` outside `planning/` and git history.

## Acceptance Criteria

- [ ] `src/tools/ipu-apps` and `ipu_app.bzl` are deleted.
- [ ] CLAUDE.md, SKILL.md and docs point at `//kernels/...`.
