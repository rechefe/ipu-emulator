# Linear layers

Twenty-eight kernels covering every matmul shape in MobileViT-S's transformer
blocks. All are **wide-vector FP32 mode only**, exact-shape matches (no
padding, no chunking, no shared shape between two kernels) — unlike softmax's
kernels, there is no boundary to sweep: each kernel handles exactly one
`(M, K, N)` (or `(K, N_OUT)`) shape and refuses everything else. All compute

```
C = A @ W^T          C[m, n] = sum_k A[m, k] * W[n, k]
```

with `W` stored output-major (`[N, K]` or `[N_OUT, K]`): row `n` holds all `K`
inputs feeding output `n`.

## Picking one

Ask the registry rather than picking by shape name:

```python
from ipu_apps.kernel_registry import resolve

resolve("matmul", shape_a=(16, 240), shape_b_t=(240, 240))
resolve("projection", k=192, n_out=576)
```

The verdict carries the app class, constructor kwargs, and — when nothing
matches — what every candidate refused and why.

## `matmul` — sixteen fixed-shape kernels

Four are layer-independent (usable for any `(M, K, N)` they happen to match);
twelve are MobileViT L3/L4/L5 projection matmuls, one per role and layer,
suffixed `_x128` (128 tokens per token-group).

| Kernel | M | K | N |
|---|---|---|---|
| `matmul_128x128` | 128 | 128 | 128 |
| `matmul_128x64x128` | 128 | 64 | 128 |
| `matmul_128x64x64` | 128 | 64 | 64 |
| `matmul_64x64x64` | 64 | 64 | 64 |
| `matmul_144x144_x128` | 256 | 144 | 144 |
| `matmul_144x288_x128` | 256 | 288 | 144 |
| `matmul_288x144_x128` | 256 | 144 | 288 |
| `matmul_432x144_x128` | 256 | 144 | 432 |
| `matmul_192x192_x128` | 64 | 192 | 192 |
| `matmul_192x384_x128` | 64 | 384 | 192 |
| `matmul_384x192_x128` | 64 | 192 | 384 |
| `matmul_576x192_x128` | 64 | 192 | 576 |
| `matmul_240x240_x128` | 16 | 240 | 240 |
| `matmul_240x480_x128` | 16 | 480 | 240 |
| `matmul_480x240_x128` | 16 | 240 | 480 |
| `matmul_720x240_x128` | 16 | 240 | 720 |

Query with `shape_a=(M, K)`, `shape_b_t=(N, K)`.

## `projection` — twelve multi-stream kernels

The all-4-pixel-stream (`P = 4`) counterparts of the single-stream `matmul_*_x128`
kernels above, one set per MobileViT layer (144/192/240) and role (`qkv`,
`outproj`, `ffn1`, `ffn2`). Each loops all 4 streams internally against one
shared weight matrix, instead of one host round-trip per stream.

| Kernel | K | N_OUT |
|---|---|---|
| `proj_qkv_144_p4` | 144 | 432 |
| `proj_outproj_144_p4` | 144 | 144 |
| `proj_ffn1_144_p4` | 144 | 288 |
| `proj_ffn2_144_p4` | 288 | 144 |
| `proj_qkv_192_p4` | 192 | 576 |
| `proj_outproj_192_p4` | 192 | 192 |
| `proj_ffn1_192_p4` | 192 | 384 |
| `proj_ffn2_192_p4` | 384 | 192 |
| `proj_qkv_240_p4` | 240 | 720 |
| `proj_outproj_240_p4` | 240 | 240 |
| `proj_ffn1_240_p4` | 240 | 480 |
| `proj_ffn2_240_p4` | 480 | 240 |

Query with `k=K`, `n_out=N_OUT` (`n_streams` defaults to 4 and rarely needs
stating explicitly).

## Data layout

`matmul_*` apps take a single `input_path`/`weights_path` pair. `proj_*_p4`
apps take `input_paths`/`output_paths` — lists of 4 paths, one per stream —
sharing one `weights_path` across all 4. Each stream's file is `[N_TG, K,
N_TOK]` float32, tg-major then channel then token; the weight file is
`[N_OUT, K]` float32, output-major.

## Using one directly

```python
from ipu_apps.kernel_registry import resolve

verdict = resolve("matmul", shape_a=(16, 240), shape_b_t=(240, 240))
app = verdict.kernel.app_class(
    inst_path="matmul_240x240_x128.bin",
    input_path="a.bin",
    weights_path="w.bin",
    output_path="c.bin",
)
state, cycles = app.run()
```
