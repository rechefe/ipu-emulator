# Linear layers

All kernels in `kernels/matmul/` and `kernels/projections/`, and
`fully_connected` in `kernels/linear/`, compute

```
C = act(A @ W^T)          C[m, n] = act(sum_k A[m, k] * W[n, k])
```

with `W` stored output-major (`[N, K]` or `[N_OUT, K]`): row `n` holds all `K`
inputs feeding output `n`. `act` is `silu` for the FFN1 kernels and the
identity for every other kernel.

## `matmul`

| Kernel | M | K | N | activation |
|---|---|---|---|---|
| `matmul_128x128` | 128 | 128 | 128 | none |
| `matmul_128x64x128` | 128 | 64 | 128 | none |
| `matmul_128x64x64` | 128 | 64 | 64 | none |
| `matmul_64x64x64` | 64 | 64 | 64 | none |
| `matmul_144x144_x128` | 256 | 144 | 144 | none |
| `matmul_144x288_x128` | 256 | 288 | 144 | none |
| `matmul_288x144_x128` | 256 | 144 | 288 | silu |
| `matmul_432x144_x128` | 256 | 144 | 432 | none |
| `matmul_192x192_x128` | 64 | 192 | 192 | none |
| `matmul_192x384_x128` | 64 | 384 | 192 | none |
| `matmul_384x192_x128` | 64 | 192 | 384 | silu |
| `matmul_576x192_x128` | 64 | 192 | 576 | none |
| `matmul_240x240_x128` | 16 | 240 | 240 | none |
| `matmul_240x480_x128` | 16 | 480 | 240 | none |
| `matmul_480x240_x128` | 16 | 240 | 480 | silu |
| `matmul_720x240_x128` | 16 | 240 | 720 | none |

All 16 are entries of `kernels/matmul/registry.toml` over 7 templates, each
run as `//kernels/matmul:<kernel>`: `matmul_small` (the four row-major kernels,
one program), and for the `_x128` kernels one template per distinct program:
`matmul_x128_2tg_c2` (`144x144`, `432x144`, `288x144`), `matmul_x128_2tg_c3`
(`144x288`), `matmul_x128_c2` (`192x192`, `240x240`, `720x240`, `384x192`,
`480x240`), `matmul_x128_c2_qkv` (`576x192`), `matmul_x128_c3` (`192x384`) and
`matmul_x128_c4` (`240x480`). A template's `activation` parameter renders the
store's `ACTIVATE.QUANTIZE`, so an FFN1 kernel shares its template's program.

Query: `bazel run //kernels:query -- matmul m=M k=K n=N [activation=silu]`. A
plain query at an FFN1 shape is refused: the activation is part of the query.

## `projection`

The single-stream `matmul_*_x128` shapes run over all 4 pixel streams in one
invocation, against one shared weight matrix. The 12 kernels are entries of
`kernels/projections/registry.toml` over three templates: `proj_p4` (one token
group, the activation a parameter), `proj_p4_2tg` and `proj_p4_2tg_silu` (two
token groups). Each runs as `//kernels/projections:<kernel>`. The family also
holds the four packed linears (`op = "packed_linear"`), which run on the
MobileViT packed layout: see
[Normalization and shaping](normalization-and-shaping.md#the-mobilevit-packed-layout).

| Kernel | K | N_OUT | activation |
|---|---|---|---|
| `proj_qkv_144_p4` | 144 | 432 | none |
| `proj_outproj_144_p4` | 144 | 144 | none |
| `proj_ffn1_144_p4` | 144 | 288 | silu |
| `proj_ffn2_144_p4` | 288 | 144 | none |
| `proj_qkv_192_p4` | 192 | 576 | none |
| `proj_outproj_192_p4` | 192 | 192 | none |
| `proj_ffn1_192_p4` | 192 | 384 | silu |
| `proj_ffn2_192_p4` | 384 | 192 | none |
| `proj_qkv_240_p4` | 240 | 720 | none |
| `proj_outproj_240_p4` | 240 | 240 | none |
| `proj_ffn1_240_p4` | 240 | 480 | silu |
| `proj_ffn2_240_p4` | 480 | 240 | none |

Query: `bazel run //kernels:query -- projection k=K n_out=N_OUT [activation=silu]`.
A query that leaves `activation` out means `activation=none`: it finds the
QKV, OutProj and FFN2 kernels, and an FFN1 kernel is found only with
`activation=silu`.

## `fully_connected`

One kernel, `kernels/linear/fully_connected`, run and tested as
`//kernels/linear/fully_connected`. It computes `y[s, :] = w @ x[s, :]` for a
fixed shape: 10 samples of 128 inputs, 64 outputs, with no activation. It is
the one kernel that runs in several element formats with one program: INT8,
INT32 (the emulator-only format: INT8 data in 32-bit elements), E4 and E5.
`y` is stored from the accumulator: INT32 for INT8 and INT32 data, FP32 for E4
and E5. The case data are seeded, generated per format inside each input's
`range`, which keeps every sum inside the accumulator. The goldens come from
the shared `linear` op (`y = linear(w, x.T).T`), rounded to the accumulator
format: exact in INT8 and INT32, `FP32_ORDER` in E4 and E5.

Query: `bazel run //kernels:query -- fully_connected`. The shape parameters
`samples`, `in_features` and `out_features` each take one value (10, 128,
64).

## Data layout

`matmul_*` kernels declare their tensors in `kernel.toml` in one of two
layouts:

- **row-major** (`matmul_64x64x64`, `matmul_128x*`): `a`, A `(M, K)`, and
  `b`, Wᵀ `(K, N)`, one XMEM row per matrix row each, and the output `y`, C
  `(M, N)`; the reference is the shared `matmul` op, `a @ b`.
- **channel-major** (`matmul_*_x128`): `b`, the data D = Aᵀ as the transformer
  blocks hold it, `(N_TG, K, N_TOK)` (`channel_major`: one row per channel and
  token group), `a`, the weights W `[N, K]` output-major
  (`output_major_weights`), and `y` `(N_TG, N, N_TOK)`, one row per output
  channel and group, the first `N_TOK` lanes valid. `M = N_TG * N_TOK`: two
  groups of 128 tokens at L3, one group of 64 or 16 at L4/L5. The reference is
  the shared `linear` op, `activation(a @ b)`, as for the projections.

A `proj_*_p4` kernel declares its tensors in `kernel.toml`, and the harness
places and packs them. `b` holds the 4 streams' data, `(N_STREAM, N_TG, K,
N_TOK)` (each stream's block channel-major). `a` holds the shared weights,
`[N_OUT, K]`, output-major. The output `y` is `(N_STREAM, N_TG, N_OUT, N_TOK)`,
stored as one XMEM row of 128 lanes per output channel, with the first `N_TOK`
lanes valid. L3 (144) has `N_TG = 2` token groups of 128; L4/L5 have
`N_TG = 1` (64 / 16 tokens padded to 128 lanes). The reference is the shared
`linear` op, `activation(a @ b)`.

`fully_connected` declares its tensors in `kernel.toml` too. `x` is
`[10, 128]`, one XMEM row per sample. `w` is `[64, 128]`, output-major; its
`transposed` pack (`kernels/linear/pack.py`) stores one row per input
feature, holding that input's weight for each of the 64 outputs. `y` is
`[10, 64]`, one accumulator vector per sample: 4 rows apart in INT8, E4 and
E5, 1 row apart in INT32.
