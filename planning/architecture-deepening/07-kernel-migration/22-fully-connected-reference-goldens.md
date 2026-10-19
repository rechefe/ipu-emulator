# `fully_connected`: goldens from the reference per format; delete the 9 `.bin`

Part of [Epic: Kernel migration](00-epic.md) · ID E07-22 · Size S · Depends on: E07-21, E03 (exact E4/E5 codecs, including the subnormal-carry fix) · Behaviour change: yes (test inputs are regenerated and goldens come from the reference instead of opaque files)

## Goal

Replace the 9 opaque `.bin` files with the E06 seeded generator and the shared
`ops.linear` reference, with goldens rounded per format (Q78). After this, every
`fully_connected` expectation can be reproduced from code in the repo.

## Background

- There is no generator for `test_data_format/{int8,fp8_e4m3,fp8_e5m2}/*.bin`
  (cluster plan 07c). Their FP8 encoding convention is unknown: it may come
  from the buggy subnormal encoder, or from OCP E4M3FN (max 448) instead of the
  repo's IEEE-style E4 (max 240) (plan 07, fact 3).
- Q78: goldens are generated from the reference per format, and the `.bin`
  files are deleted.

## Implementation

1. **Investigate before deleting.**
   - Decode the old inputs with the E03 codecs. Check for codes at or above
     `0x78`, or values above 240 in E4, which would point to OCP.
   - Run `ops.linear` on the decoded old inputs and compare it with the old
     goldens.
   - Record in the PR whether the old goldens were reproducible: exactly, within
     a tier, or not at all.
   - If they are not reproducible, that is a finding (an encoder or golden
     defect) to report, not a reason to keep the files.
2. **Replace the data.**
   - Delete the `legacy` case, `gen.py:legacy_bin` and `test_data_format/`.
   - Add `default`/`sweep` cases with the E06 generator. Use per-format input
     ranges chosen so the accumulators do not saturate (E06-08 `range`).
3. **Tiers.**
   - INT8 and INT32: `EXACT`, because the integer accumulation is exact.
   - E4 and E5: FP32 accumulation of exactly representable products, with
     `FP32_ORDER` expected. Measure it.
   - INT8 and INT32 now have their own runs, each with its own expected values
     from the same reference.

## Tests

- [ ] `bazel test //kernels/linear/fully_connected` passes every case in all 4 formats against the reference.
- [ ] The investigation results (step 1) are recorded in the PR.
- [ ] A grep finds no `.bin` under `kernels/`.

## Acceptance Criteria

- [ ] The 9 `.bin` files are deleted. Goldens come from `ops.linear`, rounded per format.
- [ ] `bazel test //...` is green.
