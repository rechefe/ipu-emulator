# conv_universal

## What it does

Runtime-parameterized **standard 3×3 convolution** (stride 1, "same" padding)
over INT8 / FP8 inputs. A single assembled binary serves any supported spatial
size and channel count — parameters are passed through CR registers, not by
recompiling.

The kernel uses a **walking-pointer / rotating-slot** scheme: the 512-byte
cyclic register holds three vertically-adjacent 128-byte chunks of one input
channel (kr=−1, kr=0, kr=+1). A single offset (`lr_walk`) walks the 9 taps of
the 3×3 window, and each new input channel rotates `lr_read` by +256 (mod 512).
Loads are issued one iteration ahead so each `mult.ve.cyclic` sees fresh data in
the same cycle. The 9 per-channel taps run in a 9-cycle body.

Quantisation: after all input channels accumulate into `r_acc` (int32),
`ACTIVATE identity → AAQ` clamps each lane to INT8 and `STR_POST_AAQ_REG`
writes the 128-byte chunk to memory.

## PyTorch / NumPy equivalent

```python
import torch
import torch.nn.functional as F

# weights: [out_ch, in_ch, 3, 3] int8; input: [in_ch, rows, cols] int8
y = F.conv2d(input.float()[None], weights.float(), bias=None, padding=1)[0]
output = torch.clamp(y.round(), -128, 127).to(torch.int8)
```

## Inputs

| Name   | Shape                       | dtype      | Layout                              | Notes |
|--------|-----------------------------|------------|-------------------------------------|-------|
| input  | `[in_ch, rows, cols]`       | INT8 / FP8 | chunk-interleaved (128 B per chunk) | `rows_per_chunk = 128 // cols` packed rows per chunk |
| kernel | `[out_ch, in_ch, 3, 3]`     | INT8 / FP8 | FPB=28 super-block packed           | see `weights.py` |

## Outputs

| Name   | Shape                  | dtype | Layout            | Notes |
|--------|------------------------|-------|-------------------|-------|
| output | `[out_ch, rows, cols]` | INT8  | chunk-interleaved | 128 B per filter per chunk |

## Constraints

- `cols ∈ {16, 32, 64}` (one packed row per mask partition group; cols=128 lives in a separate binary)
- `rows * cols ≥ 256` (at least 2 chunks)
- `in_channels ≥ 1`, `out_channels ≥ 1`
- **Not in-place** — input and output occupy separate XMEM regions

## Special notes

- **Borders:** left/right edge columns are zeroed by `mask_shift` (with
  `CR15.partition = cols` so each partition group is one packed row); the
  top/bottom off-image rows are zeroed by a **data zero region** (`cr9`) loaded
  into the off-image cyclic slot (only `local_row 0` / the last packed row reads
  it — a per-row data property a uniform mask cannot express).
- `CR15.partition` is overwritten by the harness to match `cols`.
- The `.asm` uses Jinja `{% set %}` register aliases (e.g. `{{ lr_walk }}`),
  rendered by the assembler's preprocessor before parsing.

## Tests

- `test/test_conv_universal.py` — bit-exact vs an IPU-math NumPy reference,
  parametrized over `(in_ch, out_ch, rows, cols)`. Wired into Bazel
  (`//src/tools/ipu-apps:test_conv_universal`), so it runs in CI.
