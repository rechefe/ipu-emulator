# Convolutions (2/4): merge the `conv_universal` twins into one template

Part of [Epic: Kernel migration](00-epic.md) · ID E07-17 · Size M · Depends on: E07-16 · Behaviour change: none

## Goal

Replace `conv_universal` and `conv_universal_bn_activation` with one template,
`conv_universal`, that takes `bias` and `activation` template parameters. Add 2
`registry.toml` entries that keep both names and both binaries.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/convolutions/`.

- The two `.asm` files are 0.97 similar once comments are stripped. A diff shows
  exactly 3 kinds of change, each repeated in the 3 unrolled sections:
  1. The bias seed: `MULT.EE {{ lr_kern_idx }} {{ cr_zero }} 0 ...` versus
     `MULT.EE {{ lr_kern_idx }} cr1 0 ...`. The `_bn_activation` variant
     multiplies a bias row by CR1 (1.0), and the plain variant by CR0 (0).
  2. The kernel index setup: `DEC {{ lr_kern_idx }} 1` versus
     `add {{ lr_kern_idx }} {{ lr_zero }} {{ lr_zero }}`.
  3. The activation: `ACTIVATE.QUANTIZE identity cr15` versus
     `ACTIVATE.QUANTIZE relu cr15`.
- The `app.py` files are 395 and 351 lines, each with 13 `set_cr`. Their
  domains (`conv_universal/app.py:364-367`,
  `conv_universal_bn_activation/app.py:303-320`) say: the plain variant refuses
  `apply_relu` and `has_bias`; the bn variant requires `apply_relu`, and its
  bias is optional (zeros).
- Both have `benchmark.py` files.

## Implementation

- Template `kernels/convolutions/conv_universal/`:
  - `[params]` includes `bias` (bool) and `activation` (choices `identity`,
    `relu`);
  - the program uses `{% if bias %}` for changes 1-2 and `{{ activation }}` for
    change 3.
- `registry.toml` entries:
  - `conv_universal`: `bias = false`, `activation = "identity"`;
  - `conv_universal_bn_activation`: `bias = true`, `activation = "relu"`.

  Today's query semantics are preserved by the fixed parameters: `has_bias`
  maps to `bias`, and `apply_relu` maps to `activation`. The bn entry still
  accepts `has_bias = false` with zero bias. Express this with an input `bias`
  tensor that defaults to zeros, as today.
- The reference is `ops.conv2d` with the bias and activation parameters.
- The benchmarks move. The upper-case and header rules apply (the per-family
  checklist).

## Tests

- [ ] `assemble_conv_universal` and `assemble_conv_universal_bn_activation` equal their E02 digests.
- [ ] `compare_setup` and `compare_legacy` are identical for both names.
- [ ] The routing answers for every combination of `has_bias` and `apply_relu` are unchanged. The missing combination is still refused here; E07-20 adds it.

## Acceptance Criteria

- [ ] One conv_universal program source. Both kernel names and targets are unchanged.
- [ ] `bazel test //...` is green.
