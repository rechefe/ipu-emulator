# Matmul (2/2): the `*_x128` group onto shared layouts

Part of [Epic: Kernel migration](00-epic.md) · ID E07-08 · Size M · Depends on: E07-07, E06-06 (LR-preset open question resolved) · Behaviour change: none

## Goal

Migrate the 12 `matmul_*_x128` single-stream projection matmuls onto the same
layouts as the projections (channel-major data, output-major weights), using
`registry.toml` entries over as few templates as the E02 digests allow.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/matmul/`.

- The kernels are `matmul_{144x144,144x288,192x192,192x384,240x240,240x480,288x144,384x192,432x144,480x240,576x192,720x240}_x128`.
  Their `app.py` files are 140-169 lines, with 5-8 `set_cr` and 10 `set_lr` each.
- Similarity once comments are stripped:
  - `{144x144, 432x144}` are identical, and both are 0.98 similar to `288x144`;
  - `{192x192, 240x240, 720x240}` are identical;
  - `{384x192, 480x240}` are identical, and 0.95-0.97 similar to the previous
    group;
  - `192x384` and `240x480` are 0.85-0.92 similar to the rest.

  That gives about 5 to 7 distinct programs.
- The projections family describes itself as "the all-stream (P=4) counterparts
  of the single-stream `matmul_*_x128`" (`kernels/projections/app.py:3-5`). The
  layouts match per stream.
- Oversize input is truncated today (`matmul_144x288_x128/app.py:76`
  `raw[:expected]`, and the same in `144x144`, `288x144` and `432x144`). The
  migration keeps the observable output identical for exact-size inputs.
  Rejecting oversize input is E07-09.
- The harnesses preset 10 LRs each (see E06-06's open question).

## Implementation

- One template per distinct body, with the shape as `[params]`, plus
  `registry.toml` entries with today's names. Try to merge bodies that differ
  only in constants that can become template values. Keep the merge only if the
  `assemble_<name>` digests stay equal.
- Use the shared `channel_major` and `output_major_weights` transforms (E06-05,
  made shared in E07-02).
- Shared reference `ops.matmul`. FP32 tier `FP32_ORDER` or `FP32_LONG`
  (K >= 256), measured.
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` (including LRs) and `compare_legacy` are identical for all 12 kernels.
- [ ] The MobileViT seam tests that use these kernels (`test_seam_*matmul*`) pass.

## Acceptance Criteria

- [ ] 12 entries over at most 7 templates, with the number recorded in the PR.
- [ ] `bazel test //...` is green.
