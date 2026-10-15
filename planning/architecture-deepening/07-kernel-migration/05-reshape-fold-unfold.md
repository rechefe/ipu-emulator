# Reshape (2/2): `fold` x3 and `unfold` x3

Part of [Epic: Kernel migration](00-epic.md) · ID E07-05 · Size M · Depends on: E07-04, E06-06 (LR-preset open question resolved) · Behaviour change: none

## Goal

Migrate the 6 `fold_*`/`unfold_*` kernels. These are shape-specific programs
whose bodies differ (similarity 0.63 or lower once comments are stripped), so
they stay 6 kernel folders. Do not force them into one template unless the E02
digests prove identical binaries.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/`.

- `fold_{8x8x240,16x16x192,32x32x144}`: 173-181 lines of `app.py` each, with
  10-11 `set_cr` and 7 `set_lr`. Their `.asm` headers document 4 CRs as byte
  addresses each.
- `unfold_{8x8x240,16x16x192,32x32x144}`: 147-226 lines of `app.py` each, with
  5-6 `set_cr` and 7-11 `set_lr`.
- The harnesses preset LRs, so this family depends on E06-06's open question
  (an `[abi]` LR target versus initialising LRs in the program). If LRs are
  initialised in the program instead, the binary changes. That would be a
  behaviour PR and does not belong in this issue.
- These kernels are consumed by the MobileViT seam tests
  (`test/mobilevit/test_seam_unfold_matmul_xmem_direct_l{4,5}.py`).

## Implementation

- One kernel folder each, with `kernel.toml` and the program with an upper-cased
  header. Byte-unit CRs are confirmed against the harness and marked
  `unit = "bytes"` (L16).
- Shared layouts (patch-major and token-major orderings) become family `pack.py`
  transforms, used by both `fold` and `unfold`.
- Shared reference `ops.fold`/`ops.unfold` (in `ipu_kernels.reference`, using
  `torch.nn.functional.fold`/`unfold`, or a reshape plus permute if the patch
  ordering differs; document which). Tier `EXACT`.
- The seam tests keep working through the temporary adapter from E07-02 until
  E07-24.
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` (including LRs) and `compare_legacy` for the 6 kernels are identical.
- [ ] `ops.fold(ops.unfold(x)) == x` for the three shapes. This is a reference self-check.
- [ ] The MobileViT seam tests that use these kernels pass.

## Acceptance Criteria

- [ ] The reshape family is fully migrated, and the old reshape directory is deleted.
- [ ] `bazel test //...` is green.
