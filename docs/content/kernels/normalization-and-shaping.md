# Normalization and shaping

Ten kernels covering LayerNorm, residual addition, and the spatial unfold
MobileViT-S needs between its convolutional and transformer stages. Like the
[linear layers](linear-layers.md), every kernel here is a **wide-vector FP32**,
exact-shape match — one kernel, one shape, no padding or chunking across
kernels.

## `layernorm` — four fixed-shape kernels

```
output[ch, i] = γ[ch] × (x[ch, i] − μ[i]) / σ[i] + β[ch]
```

normalizing **across channels**, independently per token (`μ`, `σ` are
per-token, not per-channel — the opposite axis from a typical batch norm).

| Kernel | Channels | Tokens |
|---|---|---|
| `layernorm_128x16` | 16 | 128 |
| `layernorm_64x192` | 192 | 64 |
| `layernorm_16x240` | 240 | 16 |
| `layernorm_256x144` | 144 | 256 |

Query with `shape=(channels, tokens)`. `input_path` is one 512-byte row per
channel (first `tokens` lanes live); `gamma_path`/`beta_path` are each a
single row holding all channel values, zero-padded to 128 lanes.

## `residual_add` — three fixed-shape kernels

Plain elementwise `C = A + B`, one whole row per channel (`tokens` lanes
live, rest zero-padded) — the same file convention as `layernorm`'s
`input_path`.

| Kernel | Tokens | Channels |
|---|---|---|
| `residual_add_16x240` | 16 | 240 |
| `residual_add_64x192` | 64 | 192 |
| `residual_add_256x144` | 256 | 144 |

Query with `shape=(tokens, channels)` — note the argument order is the
transpose of `layernorm`'s, since residual add has no reduction axis to
privilege.

## `unfold` — three fixed-shape kernels

A stride-2 space-to-depth decimation (`PixelUnshuffle(2)`'s phase
decomposition): for each of 4 output streams `s = (r_ph, c_ph)`,

```
output[s] = x[:, r_ph::2, c_ph::2]
```

| Kernel | H | W | C |
|---|---|---|---|
| `unfold_8x8x240` | 8 | 8 | 240 |
| `unfold_16x16x192` | 16 | 16 | 192 |
| `unfold_32x32x144` | 32 | 32 | 144 |

Query with `shape=(H, W, C)`. `input_path` is striped per spatial stripe and
channel, zero-padded to 128 lanes per row (see the kernel's own docstring for
the exact packing if reproducing it by hand).

## Picking one

```python
from ipu_apps.kernel_registry import resolve

resolve("layernorm", shape=(192, 64))
resolve("residual_add", shape=(64, 192))
resolve("unfold", shape=(16, 16, 192))
```

## Using one directly

```python
from ipu_apps.kernel_registry import resolve

verdict = resolve("layernorm", shape=(192, 64))
app = verdict.kernel.app_class(
    inst_path="layernorm_64x192.bin",
    input_path="x.bin",
    gamma_path="gamma.bin",
    beta_path="beta.bin",
    output_path="out.bin",
)
state, cycles = app.run()
```
