# Pooling: 5 `maxpool2d_*` kernels

Part of [Epic: Kernel migration](00-epic.md) · ID E07-14 · Size M · Depends on: E07-02, E06-11 (benchmark format) · Behaviour change: none

## Goal

Migrate the pooling family. All 5 kernels are whole-image (MemoryApp) kernels,
built on the family bases `WindowPoolApp` and `Stride2PoolApp`. They become
declared-tensor kernels with family `pack.py` transforms, and
`maxpool2d_nms7`/`nms9` become entries over one template if the digests allow.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/pooling/`.

- The per-kernel `app.py` files are 9-10 lines each. The real code is the family
  `pooling/app.py` (97 lines): `WindowPoolApp` (`:11`), `Stride2PoolApp`
  (`:40`) and `stride2_spec` (`:86`). The layout is built into the image by
  `pooling/cases.py` (54 lines).
- `maxpool2d_nms7` and `maxpool2d_nms9` are 0.98 similar once comments are
  stripped (window size).
- `maxpool2d_stride2` and `maxpool2d_stride2_tail` have `benchmark.py` and
  `results.md`.
- 4 lower-case mnemonic lines.

## Implementation

- `maxpool2d_window`, `maxpool2d_stride2` and `maxpool2d_stride2_tail` are
  folder kernels. `nms7`/`nms9` share a template with `window` as a template
  parameter if the binaries stay identical. Otherwise they stay two folders.
- The image layout from `cases.py` becomes `pack.py` transforms. Padding and
  border rows become `[scratch.*]` with `fill` where a constant is needed
  (for example -inf for max pooling, if the image builder writes one; check).
- Shared reference `op = "maxpool2d"` (`torch.nn.functional.max_pool2d`, with
  NMS as a max-pool equality mask in a local `reference.py`). Tier `EXACT`.
- The benchmarks move to the E06-11 format.
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` (the XMEM image equals the old case-built image) and `compare_legacy pooling` are identical for all 5 kernels.
- [ ] The family tests from `pooling/test.py` (101 lines) are ported or deleted with a reason.

## Acceptance Criteria

- [ ] `WindowPoolApp`, `Stride2PoolApp` and the image-building code in `pooling/cases.py` are gone.
- [ ] `bazel test //...` is green.
