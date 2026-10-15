# Convolutions (1/4): `conv1x1`, `conv3x3_relu`, `conv3x3_relu_cin1`, `conv_first_layer`

Part of [Epic: Kernel migration](00-epic.md) · ID E07-16 · Size M · Depends on: E07-02 · Behaviour change: none

## Goal

Start the convolution family, the largest one (5,197 Python lines, 4,966 asm
lines, 14 kernels, about 660 lines of lower-case mnemonics). This issue builds
the family's shared pieces and migrates the 4 kernels that need no template
merge:
- the shared layout transforms, the shared conv reference, and the
  query-geometry checks expressed as `[params]`;
- the 3 `ConvApp` whole-image kernels;
- `conv_first_layer`.

## Background

Paths are under `src/tools/ipu-apps/src/ipu_apps/kernels/convolutions/`.

- **Family `app.py`** (632 lines) holds:
  - `ConvApp(MemoryApp)` (`:69`);
  - `kernel_asm` (`:119`) and `render_asm` (`:128`);
  - `pack_input_chunked`/`unpack_output_chunked` (`:163`, `:191`);
  - `allocate_regions` (`:223`), a second placement algorithm;
  - `build_border_mask_blob` (`:266`);
  - `ConvQuery` (`:316`) and `Conv2dDescription` (`:515`), which declare the
    same 11 fields;
  - the `Conv2d` layer adapter (`:622`).
- **`conv1x1`, `conv3x3_relu`, `conv3x3_relu_cin1`** have 11-line `app.py`
  files over `ConvApp`. The image is built in `convolutions/cases.py` (181
  lines), where the weights are a prefix of the inputs because of the shared
  reseeded RNG (`cases.py:29-31`, `case_support.py:8`).
- **`conv_first_layer`** (320 lines of `app.py`, 12 `set_cr`) has a
  `benchmark.py`.
- The kernel_size/dilation/padding/stride check is copied into 11 `_supports`
  functions (cluster plan 05, A3).

## Implementation

1. **Shared pieces** (`ipu_kernels/convolutions/`):
   - `pack.py`: `chunk_interleaved` (from `pack_input_chunked` and
     `unpack_output_chunked`) and `border_mask` (a scratch `fill` generator from
     `build_border_mask_blob`).
   - `layers.py`: `LAYER_ADAPTERS = {"Conv2d": ...}` (E06-16), with one
     geometry dataclass replacing `ConvQuery` and `Conv2dDescription`.
   - The geometry check becomes `[params]` domains: `kernel_size`, `stride`,
     `padding`, `dilation` with `choices` per kernel.
   - `allocate_regions` is deleted, because placement (E06-04) owns it. Prove
     the bases are unchanged with `compare_setup`.
2. Shared reference `op = "conv2d"` (`torch.nn.functional.conv2d`, plus
   optional bias, plus optional ReLU) in `ipu_kernels.reference.ops`. FP32 tier
   measured.
3. Migrate the 4 kernels. Each folder holds `kernel.toml` and the program.
   - The mnemonics are upper-cased with `tools/upper_mnemonics.py`
     (binary-identical).
   - Cases keep the legacy correlated data through a `generator` hook, so the
     outputs stay comparable. Switching to independent streams (E06-08) is a
     follow-up commit in the same PR, after the proof. The new data may expose a
     real kernel bug. If so, triage it; do not widen the tier.
4. The `conv_first_layer` benchmark moves to the E06-11 format.

## Tests

- [ ] `compare_setup` and `compare_legacy` are identical for the 4 kernels.
- [ ] After the generator switch, all cases still pass at the measured tier. Any failure is filed as a kernel bug.
- [ ] `from_layer` for a `Conv2d` layer returns the same verdicts as before, for the conv test cases in `convolutions/test.py`.

## Acceptance Criteria

- [ ] `ConvApp`, `allocate_regions` and the copied geometry checks are gone. The 3 `ConvApp` kernels are no longer whole-image.
- [ ] `bazel test //...` is green.
