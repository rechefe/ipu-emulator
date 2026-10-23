# Add the missing conv variant: bias without ReLU

Part of [Epic: Kernel migration](00-epic.md) · ID E07-20 · Size S · Depends on: E07-17, E07-18 · Behaviour change: yes (new registry entries; queries with `has_bias=True, apply_relu=False` that are refused today now route to a kernel)

## Goal

Add the first variant that the merged templates make free: convolution with
bias and no activation. Add an entry for each template where the combination is
meaningful: `conv_universal`, `pointwise_conv_unified`, and the depthwise
template if E07-18 merged it.

## Background

- No conv kernel handles `has_bias=True, apply_relu=False` today (cluster plan 05):
  - `conv_universal/app.py:364-367` refuses bias;
  - `conv_universal_bn_activation/app.py:318-320` requires ReLU;
  - `pointwise_conv_unified/app.py:300-304` refuses ReLU and bias;
  - `pointwise_conv_unified_bn_activation` requires ReLU (`app.py:323-325`).
- After E07-17/18, the combination is one `registry.toml` entry per template:
  `bias = true`, `activation = "identity"`.

## Implementation

- Add the entries, named in the family style: `conv_universal_bias`,
  `pointwise_conv_unified_bias`, and `depthwise_conv_universal_bias` if
  applicable.
- Cost: the same as the plain variant. Nothing else claims this domain, so no
  routing ties are possible. The registry load check (E06-16) confirms this.
- Cases: `default` and a sweep over the shapes the plain variant covers. The
  reference is `ops.conv2d` with bias and no activation.
- New `assemble_*` digests are added to the E02 record in this PR, with this
  issue as the reason.

## Tests

- [ ] `resolve("conv2d", ..., has_bias=True, apply_relu=False)` returns the new entry for each template's geometry, where it used to refuse.
- [ ] The new entries pass all cases at the measured FP32 tier.
- [ ] The existing entries' routing and digests are unchanged.

## Acceptance Criteria

- [ ] Every `(has_bias, apply_relu)` combination routes to a kernel for each conv template's geometry.
- [ ] `docs/content/app-coverage.md` (or its E09 successor) lists the new entries.
- [ ] `bazel test //...` is green.
