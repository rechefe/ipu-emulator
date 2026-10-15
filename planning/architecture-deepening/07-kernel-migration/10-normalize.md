# Normalize: `layernorm_*` x4 and `l2_normalize_channels`

Part of [Epic: Kernel migration](00-epic.md) · ID E07-10 · Size M · Depends on: E07-02, E06-09 (shared `ops.layernorm`, no eps) · Behaviour change: none

## Goal

Migrate the normalize family and put its golden on the one shared LayerNorm
reference, which matches what the kernels compute and has no eps (Q57).
`l2_normalize_channels` stops being a whole-image kernel.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/normalize/`.

- **`layernorm_{128x16,16x240,64x192,256x144}`** have `app.py` files of
  153-215 lines, each with 13 `set_cr`. Similarity is 0.87-0.96 once comments
  are stripped, so the bodies are not identical. Plan 05 suggested a template
  with macros for the differing passes, which needs identical binaries to be
  proven per shape.
- **The golden** is `normalize/cases.py:26` `reference_layernorm`. Its docstring
  (`cases.py:16`) says it uses no epsilon, to match the rsqrt activation. The
  full-layer MobileViT tests use eps = 1e-5 (cluster plan 07c). Q57 picks one
  reference matching the kernel (no eps), to be revisited when AaQ is designed.
- `cases.py:61` `check_not_degenerate` guards against trivially passing data.
  Keep it as a family `test_*.py` or a generator property.
- **`l2_normalize_channels`** (14-line `app.py`) is MemoryApp. Its image is
  built in `cases.py`.

## Implementation

- `layernorm_*`:
  - Try one template with shape parameters and Jinja macros for the passes that
    differ. Keep it only if all 4 `assemble_<name>` digests match. Otherwise
    use 4 folder kernels, or 2 templates. Record which in the PR.
  - Reference `op = "layernorm"` (shared, no eps). The FP32 tier is measured,
    and `FP32_LONG` is expected for the rsqrt chain.
- `l2_normalize_channels`: declared input tensor, a family `pack.py` transform
  for the channel layout, and a local `reference.py` (`x / ||x||_2` over
  channels; state its zero-norm behaviour as the kernel computes it).
- Delete the local `reference_layernorm`. The MobileViT tests switch to
  `ops.layernorm` in E07-24. Until then they keep their own copy, and the PR
  notes it.
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` and `compare_legacy normalize` are identical for the 5 kernels.
- [ ] `ops.layernorm` reproduces the old `reference_layernorm` output bit for bit on every case input. Otherwise the max difference is recorded, and the tier still passes.
- [ ] The degenerate-data guard still fails on constant input.

## Acceptance Criteria

- [ ] There is one LayerNorm reference in the repo for kernels (`ipu_kernels.reference.ops.layernorm`), and no local copy in `kernels/`.
- [ ] No MemoryApp users remain in normalize.
- [ ] `bazel test //...` is green.
