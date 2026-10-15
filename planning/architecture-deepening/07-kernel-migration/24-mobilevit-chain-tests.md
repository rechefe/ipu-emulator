# Port the MobileViT chain tests to the harness chaining API

Part of [Epic: Kernel migration](00-epic.md) · ID E07-24 · Size M · Depends on: E07-23, E07-02, E07-05, E07-08, E07-10, E07-13, E06-15, E06-19 · Behaviour change: none

## Goal

Run the MobileViT seam and full-layer chains through `ipu_harness.chain` (E06-15)
on the migrated kernels. That removes:
- the monkeypatching and module-global reads in `fixture_kernel_support.py`;
- the 43 copied `set_cr` calls per packed chain file;
- the temporary adapters that E07-02 and later issues left for these tests;
- the local LayerNorm, SiLU and softmax reference copies.

## Background

Paths are under `src/tools/ipu-apps/test/mobilevit/`.

- `fixture_kernel_support.py:121` `skip_load_data` monkeypatches a harness
  module's `_load_data`, and lines 67-71 and 105-114 read `module.OUTPUT_BASE`
  and `module.ROW_BYTES`.
- `test_full_layer_l{4,5}_packed.py` have 43 `set_cr` calls each.
- The test sizes are 708, 695, 623 and 1,019 lines. The fixtures add 853.
- LayerNorm copies use `eps=1e-5` (`test_full_layer_l4.py`, `l5.py`). The kernel
  reference has no eps (Q57).
- Filler data is seeded with `hash((tag, "filler"))`
  (`test_full_layer_l4.py:235,375`), which depends on PYTHONHASHSEED.
- The 4 `per_stage_error` tests only print (cluster plan 07c).
- The `test_full_*` targets are `manual` and `enormous`, and take hours
  (`ipu_app.bzl:179-183`). Q73 keeps them out of the default run, in the nightly
  tier and under `--config=full`.

## Implementation

1. **Seam tests** (`test_seam_*.py`, 8 files): rewrite each as a `Chain` of 2-3
   steps with `Resident` hand-offs. They compare the final output against the
   chained shared references.
2. **Full-layer tests**:
   - Rewrite them as `Chain`s. Move them to
     `kernels/mobilevit_packed/chains/` (or the family E07-23 picks), with the
     fixtures library from E06-19.
   - Replace `hash(` seeding with `rng(tag, "filler")` (E06-08). This changes
     the data, so record the per-stage errors before and after in the PR.
   - Use `ops.layernorm` (no eps) in the chain reference too, so there is one
     semantic.
3. **`per_stage_error` tests**: assert `max_abs_err[stage] <= budget[stage]`.
   The budgets come from the measured values with 2x headroom, recorded as
   named tier overrides with a `why` (E06-10).
4. Delete `fixture_kernel_support.py`, `skip_load_data`, the temporary adapters
   and every local reference copy.

## Tests

- [ ] All seam tests pass in the default run.
- [ ] The manual full-layer tests pass when run by name, with run times and per-stage errors recorded in the PR.
- [ ] A grep finds no `monkeypatch`, `module.OUTPUT_BASE`, `hash((`, `jinja2.Template` or `set_cr(` in the chain tests.

## Acceptance Criteria

- [ ] The chains use only the public harness API (`Chain`, `Resident`, `Registry`).
- [ ] `bazel test //...` is green. The nightly tier runs the full-layer chains.
