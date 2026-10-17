# Normalization and shaping

## `layernorm`

```
output[ch, i] = γ[ch] × (x[ch, i] − μ[i]) / σ[i] + β[ch]
```

normalizing **across channels**, independently per token (`μ`, `σ` are
per-token, not per-channel — the opposite axis from a typical batch norm).

No eps (Q57): a zero variance gives `rsqrt = 0`, so a constant token column
becomes `β`. The golden is the shared `ipu_kernels.reference.ops.layernorm`
(`dim = 0`).

The four kernels are entries of `kernels/normalize/registry.toml` over the
`layernorm` template (one program per shape), each run as
`//kernels/normalize:<kernel>`.

| Kernel | Channels | Tokens |
|---|---|---|
| `layernorm_128x16` | 16 | 128 |
| `layernorm_64x192` | 192 | 64 |
| `layernorm_16x240` | 240 | 16 |
| `layernorm_256x144` | 144 | 256 |

Query: `bazel run //kernels:query -- layernorm channels=C tokens=T`. `x` and
`y` are one 128-element FP32 row per channel and token group (row
`ch * ceil(tokens / 128) + g`), tokens in the first lanes; `x` is zero past
them, and `y` holds `beta[ch]` there. `gamma` and `beta` are `channels` FP32
values in zero-padded rows. `x` sits at row 0: the program reads its base
through `CR0`.

## `l2_normalize`

```
y[ch, n] = x[ch, n] / sqrt(Σ_ch x[ch, n]²)
```

per column, down the channels. One kernel, `l2_normalize_channels`
(`//kernels/normalize/l2_normalize_channels`), takes any `channels` and
`columns`, 128 columns a tile: `bazel run //kernels:query -- l2_normalize
channels=C columns=N`. An all-zero column gives 0, not NaN: the `rsqrt`
activation maps a non-positive sum to 0.

## The MobileViT packed layout

The Layer 4 and Layer 5 transformer blocks also run in a packed layout:
`128 / tokens` channels share a row, channel `c`'s tokens at row
`c / (128 / tokens)`, lanes `(c % (128 / tokens)) * tokens` on (2 channels a
row at Layer 4, 8 at Layer 5). It is row-major over the flattened
`(channels, tokens)` tensor, so each kernel declares a packed tensor as its
rows. LayerNorm, pack, unpack and the linear use their own `op` names, so no
unpacked query reaches them; the residual adds share `residual_add` with the
unpacked ones and are told apart by `per_row`:

| Kernel | Family | Op | What |
|---|---|---|---|
| `packed_layernorm_192x64`, `packed_layernorm_240x16` | `kernels/normalize` | `packed_layernorm` | LayerNorm over the channels, `ops.layernorm` (no eps) |
| `packed_pack_192x64`, `packed_pack_240x16` | `kernels/reshape` | `pack` | one channel a row → packed |
| `packed_unpack_192x64`, `packed_unpack_240x16` | `kernels/reshape` | `unpack` | packed → one channel a row (the attention seam) |
| `packed_output_linear{,_silu}_240`, `packed_output_linear{,_silu}_192` | `kernels/projections` | `packed_linear` | one packed output row of `activation(a @ b)` a call; `k` is a parameter |
| `packed_residual_add_192x64`, `packed_residual_add_240x16` | `kernels/elementwise` | `residual_add` | the `residual_add_rows` program with `per_row` 2 and 8 |

All but the residual add load an R_MASK row (a scratch region whose bytes the
family's `gen.py` makes from `ipu_kernels.pack.r_mask`). Pack, unpack and the
linear leave R_MASK changed (LayerNorm ends by reloading all ones): the
residual add relies on R_MASK's all-ones reset value, so it needs R_MASK
reloaded first when it runs after one of them. Query, for example:
`bazel run //kernels:query -- packed_layernorm channels=240 tokens=16`.

## `residual_add`

Plain elementwise `y = a + b`. The three unpacked kernels are entries of
`kernels/elementwise/registry.toml` over the `residual_add_rows` template (one
program), each run as `//kernels/elementwise:<kernel>`. For
`residual_add_16x240` and `residual_add_64x192`, one whole row per channel
(`tokens` elements of data, the rest zero padding) — the same layout as
`layernorm`'s input; `residual_add_256x144` has `tokens > 128`, so each
channel takes `ceil(tokens / 128)` full rows.

| Kernel | Tokens | Channels |
|---|---|---|
| `residual_add_16x240` | 16 | 240 |
| `residual_add_64x192` | 64 | 192 |
| `residual_add_256x144` | 256 | 144 |

Query: `bazel run //kernels:query -- residual_add tokens=T channels=C`. Two
more entries, `packed_residual_add_{192x64,240x16}`, run the same program on
the packed layout (`per_row` 2 and 8; [above](#the-mobilevit-packed-layout)).

The CHW `residual_add` kernel (convolution stages) shares this operation but is
queried by `num_channels`; see [Convolutions](convolutions.md#residual_add).

## `unfold`

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

Each runs as `//kernels/reshape/<kernel>`. Query:
`bazel run //kernels:query -- unfold h=H w=W c=C`. The input `x` `[C, H, W]`
is laid out in spatial stripes, one row per channel and stripe (the family
`pack.py` transform `stripes`); `unfold_8x8x240`'s rows are permuted
(`paired_rows`). The output `y` `[4, C, tokens]` holds the four phases one
after another, the first `tokens` lanes of each row valid (`phase_rows`).

`fold_8x8x240`, `fold_16x16x192` and `fold_32x32x144` (op `fold`, the same
`h w c` query) are the inverses: `x` `[4, C, tokens]` in, `y` `[C, H, W]` out
in stripes. `kernels/reshape/test_fold_unfold.py` runs each pair end to end on
one model. `concat_8x8x160x160`,
`concat_16x16x128x128` and `concat_32x32x96x96` concatenate two tensors along
the channel axis, one channel per row: entries of `kernels/reshape/registry.toml`
over the `concat` template, each run as `//kernels/reshape:<kernel>`. Query:
`bazel run //kernels:query -- concat h=H w=W c_a=C_A c_b=C_B`.
