# Attention (1/2): `qk_scores` x3 and `attn_scores_km` x3

Part of [Epic: Kernel migration](00-epic.md) · ID E07-12 · Size M · Depends on: E07-11, E06-06 (LR-preset open question resolved) · Behaviour change: none

## Goal

Migrate the 6 score kernels of the attention family. The family's
datapath-mirror goldens move to the shared `datapath` module, and each one is
paired with a math reference.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/attention/`.

- `qk_scores_{16x60,64x48,256x36}` and `attn_scores_km_{16x60,64x48,256x36}`
  have `app.py` files of 170-237 lines each, with 5-9 `set_cr` and 9-11
  `set_lr`. Similarity once comments are stripped is 0.65-0.85, so they stay 6
  folder kernels.
- The family `attention/app.py` (172 lines), `cases.py` (295) and `test.py`
  (784) are large. `cases.py` holds `agg_fold`/`acc_fold`, which mirror the
  emulator's float64/float32 fold rounding and are checked at rtol = atol = 0.
- `attention/test.py:723` `needs_softmax` is dead: every `.asm` is data, so
  the skip never triggers (cluster plan 06).
- The `.asm` headers of `qk_scores_256x36` and `attn_scores_km_256x36` discuss
  byte sizes. Confirm the CR units.

## Implementation

- One folder kernel each, with `kernel.toml` and an upper-cased program header
  that holds the design notes from `app.py`.
- Goldens:
  - the primary is `op = "matmul"` (scores = Q K^T, scaled as the kernel
    computes) at an FP32 tier;
  - `datapath.agg_fold`/`acc_fold` (moved to
    `ipu_kernels/reference/datapath.py`) stay as an additional `EXACT` check in
    a family `test_datapath.py`. A bug shared by the mirror and the emulator is
    still caught by the math reference (E06-09 rule).
- Delete `needs_softmax`. Split the family tests that go beyond case x format
  into `test_*.py` files of under 600 lines each (Q66).
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` (including LRs) and `compare_legacy` are identical for the 6 kernels.
- [ ] The datapath mirror and the math reference both pass for every case.
- [ ] The MobileViT seam test `test_seam_qkv_matmul_scores_km_64x48` passes.

## Acceptance Criteria

- [ ] The 6 score kernels are migrated, and `agg_fold`/`acc_fold` live only in `ipu_kernels.reference.datapath`.
- [ ] `bazel test //...` is green.
