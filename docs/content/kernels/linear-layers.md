# Linear layers

All kernels in `kernels/matmul/` and `kernels/projections/` compute

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

Query: `matmul shape_a=M,K shape_b_t=N,K [activation=silu]`.

## `projection`

The single-stream `matmul_*_x128` shapes run over all 4 pixel streams in one
invocation, against one shared weight matrix.

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

Query: `projection k=K n_out=N_OUT [activation=silu]`.

## Data layout

Every kernel takes `input_path`, `weights_path` and `output_path`. The weight
file is `[N, K]` (or `[N_OUT, K]`) float32, output-major, stored verbatim.

`matmul_*` kernels use one of two layouts (each kernel's `cases.py` states its
own):

- **row-major** (`matmul_64x64x64`, `matmul_128x*`): the input file is A,
  `(M, K)`; the output file is dense C, `(M, N)`.
- **channel-major** (`matmul_*_x128`): the input file is D = Aᵀ, `(K, M)`, as
  the transformer blocks hold it; the output file is raw XMEM rows,
  `(N_TG, N, 128)`, with the first `N_TOK` lanes of each row valid.

`proj_*_p4` kernels take the 4 streams back to back in **one** input file,
`(N_STREAM, N_TG, K, N_TOK)` float32 (each stream's block channel-major), and
write one output file of raw XMEM rows, `(N_STREAM, N_TG, N_OUT, 128)`, with the
first `N_TOK` lanes of each row valid. L3 (144) has `N_TG = 2` token groups of
128; L4/L5 have `N_TG = 1` (64 / 16 tokens padded to 128 lanes).
