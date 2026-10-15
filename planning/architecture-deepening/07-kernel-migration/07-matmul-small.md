# Matmul (1/2): the four small matmuls as one template

Part of [Epic: Kernel migration](00-epic.md) · ID E07-07 · Size S · Depends on: E07-02 · Behaviour change: none

## Goal

Migrate `matmul_128x128`, `matmul_128x64x128`, `matmul_128x64x64` and
`matmul_64x64x64`. Their bodies are identical once comments are stripped, so
they become one template plus 4 `registry.toml` entries.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/matmul/`.

- The 4 bodies hash the same once comments are stripped. Their `app.py` files
  are 153-162 lines, each with 10 `set_cr` and no LR presets. Their `.asm`
  headers discuss byte sizes in several places. Confirm the CR units against the
  harness.
- The family `matmul/app.py` (127 lines) and `cases.py` (67) hold the query and
  the shared cases.

## Implementation

- Template `kernels/matmul/matmul_small/`, with `[params]` `m`, `k` and `n` as
  today's shapes allow. Fixed values per entry come from `registry.toml`.
- The layouts reuse the projections transforms where they match. Otherwise add
  family `pack.py` transforms.
- Shared reference `ops.matmul`. The FP32 tier is measured (`FP32_ORDER` for
  K = 64/128 is expected).
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` and `compare_legacy` are identical for the 4 kernels.
- [ ] Routing through `//kernels:query` is unchanged for these shapes.

## Acceptance Criteria

- [ ] 4 entries over 1 template. No per-shape folders remain for these kernels.
- [ ] `bazel test //...` is green.
