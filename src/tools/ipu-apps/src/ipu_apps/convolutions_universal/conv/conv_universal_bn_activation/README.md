# conv_universal_bn_activation

## What it does

The **bn_activation twin** of [`conv_universal`](../conv_universal/README.md): a
standard 3×3 convolution (stride 1, "same" padding) with a **folded
per-output-channel bias** and a **ReLU activation**, quantised to INT8.

BatchNorm is expected to be already folded into the weights and bias upstream
(no separate BN op). The per-filter INT8 bias is packed as byte 0 of each FPB=28
super-block and seeded into `r_acc` via `mult.ve.padded` + `acc.first` (so the
accumulator starts at `bias`); the 9 conv taps accumulate on top. At the channel
boundary, `ACTIVATE relu → AAQ` applies ReLU and clamps to INT8, and
`STR_POST_AAQ_REG` writes the 128-byte chunk.

The convolution core (walking pointer, rotating cyclic slots, 9-cycle body) is
identical to `conv_universal`.

## PyTorch / NumPy equivalent

```python
import torch
import torch.nn.functional as F

# weights: [out_ch, in_ch, 3, 3] int8; bias: [out_ch] int8; input: [in_ch, rows, cols] int8
y = F.conv2d(input.float()[None], weights.float(), bias=None, padding=1)[0]
y = y + bias.float()[:, None, None]
output = torch.clamp(F.relu(y).round(), -128, 127).to(torch.int8)
```

## Inputs

| Name   | Shape                   | dtype      | Layout                    | Notes |
|--------|-------------------------|------------|---------------------------|-------|
| input  | `[in_ch, rows, cols]`   | INT8 / FP8 | chunk-interleaved (128 B) | `rows_per_chunk = 128 // cols` |
| kernel | `[out_ch, in_ch, 3, 3]` | INT8 / FP8 | FPB=28 super-block packed | see `weights.py` |
| bias   | `[out_ch]`              | INT8       | byte 0 of each super-block| folded BN bias |

## Outputs

| Name   | Shape                  | dtype | Layout            | Notes |
|--------|------------------------|-------|-------------------|-------|
| output | `[out_ch, rows, cols]` | INT8  | chunk-interleaved | 128 B per filter per chunk; ReLU + clamp applied |

## Constraints

- `cols ∈ {16, 32, 64}` (one packed row per mask partition group)
- `rows * cols ≥ 256` (at least 2 chunks)
- `in_channels ≥ 1`, `out_channels ≥ 1`
- **Not in-place** — input and output occupy separate XMEM regions

## Special notes

- **Borders:** left/right edge columns are zeroed by `mask_shift` (with
  `CR15.partition = cols`). The top/bottom off-image rows are zeroed by mask
  slots, with **no zero region**: a single R_MASK blob (loaded once at init)
  carries slot 0 (none), slot 3 (zero packed row 0, used by the top/g0 section)
  and slot 6 (zero the last packed row, used by the bottom/gN section). No
  mid-program R_MASK reload.
- Bias is folded — expects BatchNorm already absorbed into weights/bias before
  packing. Activation is ReLU; `AAQ` then clamps to INT8.
- `CR15.partition` is overwritten by the harness to match `cols`.
- The `.asm` uses Jinja `{% set %}` register aliases.

## Tests

- `test/test_conv_universal_bn_activation.py` — bit-exact vs an IPU-math NumPy
  reference (conv + bias + ReLU + clamp). Wired into Bazel
  (`//src/tools/ipu-apps:test_conv_universal_bn_activation`); runs in CI.
- `test/test_conv_universal_bn_activation_pytorch.py` — bit-exact vs PyTorch
  INT8 `conv2d + bias + relu + clamp`. **Local-only cross-check** (requires
  `torch`, not installed in CI); not a Bazel target.
