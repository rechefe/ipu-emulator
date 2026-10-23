# Create `ipu_harness` and `ipu_kernels` packages, Bazel libraries and import rules

Part of [Epic: Kernel framework](00-epic.md) · ID E06-01 · Size S · Depends on: E03 (`ipu_isa` package exists), E01 (import-graph test mechanism) · Behaviour change: none

## Goal

Create the two packages the framework is built in, with empty modules, Bazel
targets, and an import rule enforced by a test. Nothing uses them yet, so the
issue is mergeable on its own.

## Background

- Q69 lists `ipu_harness` and `ipu_kernels` among the packages, one directory
  each under `src/tools/`. Q62 requires one-way dependencies, enforced by Bazel
  deps and an import-graph test, with no lazy imports used to dodge cycles.
- Today's framework code lives in `src/tools/ipu-apps/src/ipu_apps/kernel_registry/`
  (2,145 lines). It has a base↔registry cycle broken by lazy imports
  (`base.py:58`, `base.py:95`, `memory.py:54`, `kernels/softmax/app.py:91`).
- Q80 names the kernel label `//kernels/<family>/<name>`.

## Implementation

1. Decide the location (open question below). These drafts assume a top-level
   `kernels/` tree of Bazel packages, plus:
   - `src/tools/ipu-harness/src/ipu_harness/`: the framework. The modules are
     added by later issues: `manifest.py`, `params.py`, `placement.py`,
     `packing.py`, `abi.py`, `assembly.py`, `cases.py`, `reference.py`,
     `tolerance.py`, `runner.py`, `stages.py`, `registry.py`, `testing.py`.
   - `src/tools/ipu-kernels/src/ipu_kernels/`: shared kernel-side Python.
     `reference/` holds shared references by op. `pack/` holds named pack
     transforms shared across families. `<family>/` holds family code that is
     neither a kernel nor a framework concern.
2. Bazel:
   - `//src/tools/ipu-harness:ipu_harness_lib` depends on `ipu_isa`, `ipu_as`
     (session API only) and `ipu_emu` (reference-model API only).
   - `//src/tools/ipu-kernels:ipu_kernels_lib` depends on `ipu_harness_lib`.
   - Add `kernels/BUILD.bazel` (empty package root) and
     `kernels/defs.bzl` (empty `ipu_kernel` macro stub; filled in E06-13).
3. Import rules, added as rows of `tools/import_graph.toml` (`//tools:test_import_graph`, E03-02). The `ipu_kernels.reference` rule and the "not at module level" rules are finer than a row: extend the table's schema (for example dotted keys matched by longest prefix) and the test here:
   - `ipu_harness` may import `ipu_isa`, `ipu_as` and `ipu_emu`. It must not
     import `ipu_apps`, `ipu_kernels`, `ipu_debug` at module level, or `torch`
     at module level. The reference is loaded lazily by the checker (E06-09)
     because torch is test-only.
   - `ipu_kernels` may import `ipu_harness`. Shared references in
     `ipu_kernels.reference` must not import `ipu_emu` (a reference that
     imports the emulator is a copy, not a reference).
   - Nothing imports `ipu_kernels` except kernel tests and the registry loader.
4. `ruff` and strict typing on `ipu_harness` from day one (Q65).

## Tests

- [ ] `bazel build //src/tools/ipu-harness/... //src/tools/ipu-kernels/... //kernels/...` succeeds.
- [ ] The import-graph test fails on a planted `import ipu_apps` in `ipu_harness`, and on a planted `import ipu_emu` in `ipu_kernels.reference`.
- [ ] `ruff check` and the type check pass on the new packages.

## Acceptance Criteria

- [ ] Both packages exist with Bazel libraries and module docstrings that state each module's responsibility (no README).
- [ ] The import rules above are enforced by a test in `bazel test //...`.
- [ ] `bazel test //...` is green.

## Open question

- Kernel tree location: (a) top-level `kernels/` (the label is exactly
  `//kernels/<family>/<name>`, as in Q80), or (b)
  `src/tools/ipu-kernels/kernels/<family>/<name>` (as in Q69, with a longer
  label). The rest of E06 and E07 is written against (a).
