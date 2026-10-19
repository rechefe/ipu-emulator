# Convolutions (3/4): merge the pointwise and depthwise twins

Part of [Epic: Kernel migration](00-epic.md) · ID E07-18 · Size M · Depends on: E07-17 · Behaviour change: none

## Goal

Apply the E07-17 pattern to the two remaining twin pairs:
`pointwise_conv_unified` / `pointwise_conv_unified_bn_activation` and
`depthwise_conv_universal` / `depthwise_conv_universal_bn_activation`. Each
pair becomes one template with `bias` and `activation` parameters, plus 2
`registry.toml` entries.

As landed: each pair is one template, `kernels/convolutions/pointwise_conv_tpl/`
and `kernels/convolutions/depthwise_conv_tpl/` (`<name>_tpl`, as E07-17), with
E07-17's one parameter set: no `bias`/`activation` parameters, the programs read
the query's `has_bias` and `apply_relu`. Both pairs merge with identical
binaries (2 program sources for the 4 kernels). The pointwise seed follows
`apply_relu` (its bias region and CR10 follow the seed, and CR10's other
binding would need an `and` condition, E07-20); the depthwise seed is
`has_bias or apply_relu`, and its `fpb` (CR7) is a parameter each entry fixes.
The old `depthwise_conv_universal` folder stays frozen as stage 1 of the old
stride-2 kernels until E07-19 (IMPLEMENTATION-DECISIONS, E07-18).

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/convolutions/`.

- **Pointwise twins:** 0.94 similar once comments are stripped. Their `app.py`
  files are 325 and 347 lines, 13 `set_cr` each. The domains are at
  `pointwise_conv_unified/app.py:300-304` and
  `pointwise_conv_unified_bn_activation/app.py:301-325`.
- **Depthwise twins:** 0.89 similar, so there are more differences than the 3
  conv_universal ones. Their `app.py` files are 365 and 329 lines. The domains
  are at `depthwise_conv_universal/app.py:337-339` and
  `depthwise_conv_universal_bn_activation/app.py:301-303`. The bn variant
  requires both bias and ReLU.
- All four have `benchmark.py` files.
- `depthwise_conv_universal` is also stage 1 of the 3
  `depthwise_conv_stride2_*` kernels (E07-19). Its program must stay
  addressable as a stage.

## Implementation

- Diff each pair and list every difference in the PR. A difference becomes a
  template parameter only if it is an expression of `bias`/`activation`. Any
  other difference becomes a named parameter with a comment. If the depthwise
  pair does not merge with identical binaries, keep it as 2 folder kernels and
  record why. Do not change binaries in this issue.
- `registry.toml` entries keep all four names.
- References: `ops.conv2d` with `groups = channels` for depthwise.
- Apply the per-family checklist.

## Tests

- [ ] All four `assemble_*` digests equal E02.
- [ ] `compare_setup` and `compare_legacy` are identical for all four.
- [ ] The routing answers for every `has_bias`/`apply_relu` combination are unchanged.

## Acceptance Criteria

- [ ] There are at most 2 program sources for the 4 kernels, or a recorded reason for keeping 3-4.
- [ ] `bazel test //...` is green.
