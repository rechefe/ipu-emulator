# Softmax: 5 kernels and their benchmarks

Part of [Epic: Kernel migration](00-epic.md) · ID E07-11 · Size M · Depends on: E07-02, E06-11 (benchmark format decided) · Behaviour change: none

## Goal

Migrate `softmax_rows`, `softmax_rows_long`, `softmax_rows_partial`,
`softmax_columns` and `softmax_columns_packed`, together with their benchmarks,
their layer adapter and the softmax conformance suite. Attention depends on this
family (`FAMILY_DEPS = {"attention": ["softmax"]}`), so it goes first.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/`.

- The family base `SoftmaxApp` (`softmax/app.py:77-190`) allocates rows, writes
  a resident log2(e) row and sets CRs in `setup()` (`app.py:143-169`). It
  guards its own spec through the reflection lookup (`app.py:91,108`). It
  registers a layer adapter (`app.py:316` `_softmax_layer`, via
  `@register_layer`).
- Each kernel has a `benchmark.py` and a `results.md`. `softmax_rows_partial`
  also has a `STATUS.md`.
- Comment drift: `softmax_rows/softmax_rows.asm:32` says `CR7 = ROW_BYTES (512)`,
  but the harness writes `set_cr(7, 1)`, in rows (`app.py:166`). The header of
  `softmax_columns.asm` documents byte-address CRs. Confirm each against the
  harness.
- About 24 lines of lower-case mnemonics in this family.
- `test/test_kernel_registry.py` runs a softmax conformance suite against
  whichever softmax kernel the registry resolves (`BUILD.bazel:40-43`).

## Implementation

- One folder kernel each. The `kernel.toml` holds:
  - the shape and dim parameters, as the domain;
  - a `[scratch.log2e]` region with `fill = 1.4426950408889634`, the resident
    row;
  - the scratch rows that `_alloc` reserves today, as `[scratch.*]`.
- The `[abi]` table replaces `_crs()`. CR7 is documented as rows in the `.asm`
  header, which fixes the drift.
- Shared reference `op = "softmax"`. Tier `TRANSCENDENT` (provisional, AaQ).
- The layer adapter moves to `ipu_kernels/softmax/layers.py` as
  `LAYER_ADAPTERS` (E06-16). The conformance suite moves to
  `kernels/softmax/test_conformance.py` and uses `Registry.resolve`.
- The benchmarks move to the E06-11 format. `results.md` and `STATUS.md` notes
  go into the `.asm` header (Q79) or stay as benchmark output; nothing else
  remains.
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` (including the log2(e) row) and `compare_legacy softmax` are identical for the 5 kernels.
- [ ] The conformance suite passes against the new registry.
- [ ] `from_layer` for a softmax layer returns the same verdict as before.
- [ ] The benchmark targets run and write results.

## Acceptance Criteria

- [ ] No `SoftmaxApp`, `_harness_spec` use or constructor guard remains.
- [ ] `bazel test //...` is green.
