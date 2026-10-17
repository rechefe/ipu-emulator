# Delete MemoryApp and the whole-image path

Part of [Epic: Kernel framework](00-epic.md) · ID E06-20 · Size S · Depends on: E07-04, E07-10, E07-14, E07-15, E07-16 (the 15 MemoryApp kernels migrated) · Behaviour change: none

## Goal

Remove the "caller supplies a complete pre-built XMEM image" contract (Q83).
After this issue, every region is placed and packed by the framework.
Raw-image inspection happens in the debugger's XMEM view, not through kernel
input files.

## Background

- `kernel_registry/memory.py:1-5`: "The input file contains every row before
  the output region, including weights, constants, padding and scratch.
  Layout preparation belongs to cases/callers."
- 15 kernels use it, directly through `memory_spec` or through the family bases
  `ConvApp` (`convolutions/app.py:69`), `WindowPoolApp`/`Stride2PoolApp`
  (`pooling/app.py:11,40`) and `DescriptorApp` (`sampling/app.py:6`):
  - `conv1x1`, `conv3x3_relu`, `conv3x3_relu_cin1`
  - `channel_peak`, `score_threshold`
  - `l2_normalize_channels`
  - `maxpool2d_nms7`, `maxpool2d_nms9`, `maxpool2d_stride2`,
    `maxpool2d_stride2_tail`, `maxpool2d_window`
  - `depth_to_space`, `identity`
  - `sample_descriptors`, `sample_descriptors_separable`

  Q77 counted 16. Re-count at implementation time.
- Q77b: the user did not see why these kernels are special. They are not
  special. Their cases build the image that a framework would otherwise build.

## Implementation

- Confirm with a grep that no `kernels/` or `ipu_kernels` code imports
  `memory.py`, `MemoryApp`, `MemoryLayout` or `memory_spec`. E07 removes the
  last users.
- Delete `kernel_registry/memory.py`, and remove its names from
  `kernel_registry/__init__.py`.
- Delete helpers used only by the image builders in the migrated families'
  `cases.py` files, if they are still present.
- Docs: remove the MemoryApp sections in `docs/content/adding-applications.md`
  and `building-applications.md`. Point raw-image debugging at the debugger XMEM
  view (E08).

## Tests

- [ ] `bazel test //...` is green with `memory.py` gone.
- [ ] A grep test: no `MemoryApp`, `memory_spec` or `MemoryLayout` in the repo outside the history and planning docs.

## Acceptance Criteria

- [ ] The whole-image path no longer exists. Every kernel's inputs are declared tensors in `kernel.toml`.
