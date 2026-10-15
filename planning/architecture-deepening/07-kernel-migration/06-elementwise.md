# Elementwise: `residual_add` x4

Part of [Epic: Kernel migration](00-epic.md) · ID E07-06 · Size S · Depends on: E07-02, E06-11 (benchmark open question) · Behaviour change: none

## Goal

Migrate the elementwise family. The three shape-named `residual_add_*` kernels
have identical bodies, so they become `registry.toml` entries over one template.
The generic `residual_add` stays a folder kernel with its benchmark.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/elementwise/`.

- `residual_add_{16x240,64x192,256x144}` have identical bodies once comments are
  stripped. Their `app.py` files are 95-125 lines, each with 8 `set_cr`.
- `residual_add` (165 lines of `app.py`, 5 `set_cr`) is parametric over
  `num_channels`. It guards its own spec in the constructor
  (`residual_add/app.py:99` `SPEC.guard`), which E06-03 makes unnecessary. It
  has a `benchmark.py`.
- A few lower-case mnemonic lines remain in the family.
- The `scale` acceptance kernel (E06-13) already lives in `kernels/elementwise/`.

## Implementation

- Template `kernels/elementwise/residual_add_fixed/` (an open name: pick one
  that does not collide with the generic kernel), plus 3 `registry.toml`
  entries with today's names.
- `residual_add` folder kernel: its `[params]` domain replaces the constructor
  guard, and its benchmark configs move to the E06-11 benchmark format.
- Shared reference `ops.residual_add` (`a + b`). Tier `FP32_ORDER` or `EXACT`,
  after measurement (a single FP32 add is expected to be `EXACT`).
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` and `compare_legacy elementwise` are identical for the 4 kernels.
- [ ] The benchmark target produces a results table equivalent to today's `results.md`.

## Acceptance Criteria

- [ ] The elementwise family holds `scale`, `residual_add`, one template and a `registry.toml`. No old harness files remain.
- [ ] `bazel test //...` is green.
