# Pilot: projections in INT8 and E4

Part of [Epic: Kernel migration](00-epic.md) · ID E07-03 · Size M · Depends on: E07-02, E04 (format-independent handlers, Q48; CR scalars in the active format, Q49), E03 (INT8/E4 codecs) · Behaviour change: yes (adds INT8 and E4 runs and goldens for 12 kernels; FP32 unchanged)

## Goal

Show that a migrated family runs unchanged in more than one element format
(Q44, Q50). Add `int8` and `e4` to the projection templates' `formats`, measure
the error against the reference, and fix the tolerance tiers for these formats.
This issue is the template for every later "add formats to family X" follow-up.

## Background

- Every kernel today runs in wide FP32 only (`ExecutionConfig(mode="fp32")`,
  `kernels/projections/app.py:382`).
- Q44: kernels are not wide-only, and any kernel must be simulable in FP8.
- Q50: 128 elements per vector, and only the width changes. Layouts pack per
  format, goldens are per format, and the pilot runs FP32, INT8 and E4.
- Q49: CR scalars are decoded in the active format. Projections pass only
  integers and bases in CRs, so they are not affected. `NEG_ONE = -1` and
  `FULL_BOUND = 126` are counts compared against LRs.
- The emulator multiplies FP8/INT8 elements into an FP32/INT32 accumulator.
  SiLU in `ACTIVATE.QUANTIZE` is the AaQ placeholder (Q25, Q35), so the E4
  SiLU error bound is provisional.

## Implementation

1. Add `formats = ["fp32", "int8", "e4"]` to the three template `kernel.toml`
   files.
2. Check that the program is truly format-independent:
   - Region bases come in rows.
   - No byte strides are baked into the `.asm`.
   - The output store type matches the declaration. If the output is stored via
     `ST.AAQ`, then `store = "element"`. If it is stored via `ST.ACC`, then
     `store = "accumulator"`, and the INT8 output takes 4 rows per vector
     (E06-04).
   - If any of this fails, fix the kernel in a separate behaviour PR first, and
     prove FP32 is unchanged with the E02 digests.
3. Cases for INT8 and E4 use the E06 default generator, not the legacy FP32
   generator. FP32 keeps the legacy one, so its output digests stay valid.
   - The projections store through `STR_POST_AAQ_REG` (all 12 programs), so
     the output is an element tensor. In INT8, a K = 192 dot product of
     full-range INT8 values saturates almost every output, so the test would
     pass trivially.
   - Set a per-format input range in `kernel.toml`, for example
     `range = { int8 = [-3, 3], e4 = [-1, 1] }`. Check that fewer than 1% of
     the reference outputs saturate, and record that fraction in the PR.
4. Measure the max abs/rel error per case and format against `ops.linear`, with
   the result rounded to the output format. Choose tiers:
   - INT8: expected `INT8_LSB` for identity, provisional for SiLU.
   - E4: expected `FP8_ULP` for identity, provisional for SiLU.
   Record the measurements in the tier docstrings (E06-10). This also settles
   E06-10's open question on the tier set.
5. Add a short section to the E06 how-to (E06-21): "Adding a format to a
   migrated kernel", with this issue as the worked example.

## Tests

- [ ] `bazel test //kernels/projections/...` runs 12 kernels x 3 formats. Every run passes its tier.
- [ ] The FP32 runs still match the E02 output digests, so `compare_legacy` is unchanged.
- [ ] A planted wrong `pack` for INT8 (FP32 element size) fails with a size error, not a wrong answer.

## Acceptance Criteria

- [ ] The projections pass in FP32, INT8 and E4. The tiers per format are measured and recorded.
- [ ] The follow-up "add formats" issues for later families can copy this issue's steps.
- [ ] `bazel test //...` is green, within the CI budget. If INT8/E4 for 12 kernels exceeds it, move the non-default cases to `tier = "nightly"` (Q73).

## Risks

- SiLU through the AaQ placeholder in E4/INT8 may produce errors no stable tier
  can describe. In that case, keep FFN1 entries `formats = ["fp32", "int8"]`
  or FP32 only, and record an open question for the AaQ work (Q35).
