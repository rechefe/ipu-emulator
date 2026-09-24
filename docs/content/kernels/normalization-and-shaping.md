# Normalization and shaping

## `layernorm`

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

Query: `layernorm shape=channels,tokens`. `input_path` is one 128-element FP32 row
per channel (and token group), tokens in the first lanes and zero padding
after; `gamma_path`/`beta_path` each hold the `channels` FP32 values
(`layernorm_128x16` takes them as one zero-padded 128-element row).

## `residual_add`

Plain elementwise `C = A + B` (`input_a_path`, `input_b_path`). For
`residual_add_16x240` and `residual_add_64x192`, one whole row per channel
(`tokens` lanes live, rest zero-padded) — the same file convention as
`layernorm`'s `input_path`; `residual_add_256x144` has `tokens > 128`, so it
flattens the problem into `ceil(tokens / 128) * channels` full rows.

| Kernel | Tokens | Channels |
|---|---|---|
| `residual_add_16x240` | 16 | 240 |
| `residual_add_64x192` | 64 | 192 |
| `residual_add_256x144` | 256 | 144 |

Query: `residual_add shape=tokens,channels` (the transpose of `layernorm`'s
order).

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

Query: `unfold shape=H,W,C`. `input_path` is striped per spatial stripe and
channel, zero-padded to 128 lanes per row; `unfold_8x8x240`'s rows are
permuted by its `pack_input_rows`.

`fold_8x8x240`, `fold_16x16x192` and `fold_32x32x144` (op `fold`, same
`shape=(H, W, C)` query) are the inverses. `concat_8x8x160x160`,
`concat_16x16x128x128` and `concat_32x32x96x96` (op `concat`,
`shape=(H, W, C_A, C_B)`) concatenate two tensors along the channel axis.
