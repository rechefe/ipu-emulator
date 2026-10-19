# Sampling and detect: 4 whole-image kernels

Part of [Epic: Kernel migration](00-epic.md) · ID E07-15 · Size M · Depends on: E07-02 · Behaviour change: none

## Goal

Migrate `sample_descriptors`, `sample_descriptors_separable`, `channel_peak`
and `score_threshold`. All four are whole-image (MemoryApp) kernels, and they
become declared-tensor kernels.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/`.

- **sampling**:
  - `sample_descriptors` and `sample_descriptors_separable` have 10-line
    `app.py` files over the family base `DescriptorApp` (`sampling/app.py:6`).
  - The image is built by `sampling/cases.py` (179 lines).
  - Family `test.py`: 211 lines.
- **detect**:
  - `channel_peak` (17 lines) and `score_threshold` (18 lines) use
    `memory_spec` directly.
  - Each has its own `cases.py` that builds the image.
  - There is no family-level Python, and 2 lower-case mnemonic lines.
- The two families are small (582 Python lines together) and share nothing
  except the MemoryApp pattern, so one issue covers both.

## Implementation

- One folder kernel each. The inputs are the logical tensors the cases start
  from (feature maps, sample coordinates, score maps). The image layouts become
  family `pack.py` transforms (`ipu_kernels/sampling/pack.py`,
  `ipu_kernels/detect/pack.py`).
- References: local `reference.py` files. Descriptor sampling uses
  `torch.nn.functional.grid_sample` only if it matches the kernel's
  interpolation exactly; otherwise write explicit indexing and state the
  interpolation in the docstring. `channel_peak` is an argmax/max over
  channels. `score_threshold` is a comparison mask. Tier `EXACT` where the
  outputs are selections, measured otherwise.
- Apply the per-family checklist.

## Tests

- [ ] `compare_setup` and `compare_legacy` are identical for all 4 kernels.
- [ ] The sampling family tests are ported to `test_*.py` or deleted with a reason.

## Acceptance Criteria

- [ ] `DescriptorApp` and the image builders are gone. No MemoryApp users remain in sampling or detect.
- [ ] `bazel test //...` is green.
