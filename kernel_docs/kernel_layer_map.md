# Kernel → layer map

Which MobileViT transformer layer each app in `src/tools/ipu-apps/src/ipu_apps/`
belongs to. The on-disk layout is intentionally **flat** — kernels are *not*
grouped into `L3/`/`L4/`/`L5/` directories, because each kernel is merged to
master individually and a flat `ipu_apps.<kernel>` import path keeps every
cherry-pick self-contained. This file is the grouping.

All 39 apps run in wide-vector FP32 debug mode. `bazel test //...` →
**50/50 targets, 389 test cases** as of 2026-08-04.

---

## Layer 3 — d=144, N=256 tokens (2 token groups × 128), h=4, head_dim=36

| Kernel | Role |
|---|---|
| `unfold_32x32x144` | space-to-depth, 4 stride-2 streams |
| `layernorm_256x144` | layer norm |
| `residual_add_256x144` | residual add |
| `qk_scores_256x36` | QKᵀ scores, **query-major** |
| `attn_v_256x36` | attn@V, query-major P, uses **AGG** |
| `attn_scores_km_256x36` | QKᵀ scores, **key-major** |
| `attn_v_bcast_36` | attn@V, key-major P, broadcast (**ACC.ADD**, no AGG) |
| `matmul_144x144_x128` | projection |
| `matmul_288x144_x128` | projection |
| `matmul_432x144_x128` | projection |
| `matmul_144x288_x128` | projection |

## Layer 4 — d=192, N=64 tokens/stream, P=4, h=4, head_dim=48, L=4

| Kernel | Role |
|---|---|
| `unfold_16x16x192` | space-to-depth, 4 stride-2 streams |
| `layernorm_64x192` | layer norm |
| `residual_add_64x192` | residual add |
| `qk_scores_64x48` | QKᵀ scores, **query-major** |
| `attn_v_64x48` | attn@V, query-major P, uses **AGG** |
| `attn_scores_km_64x48` | QKᵀ scores, **key-major** |
| `attn_v_bcast_48` | attn@V, key-major P, broadcast (**ACC.ADD**, no AGG) |
| `matmul_192x192_x128` | projection (single token group) |
| `matmul_384x192_x128` | projection (single token group) |
| `matmul_576x192_x128` | projection (single token group) |
| `matmul_192x384_x128` | projection (single token group) |

## Layer 5 — d=240, N=16 tokens/stream, P=4, h=4, head_dim=60, L=3

| Kernel | Role |
|---|---|
| `unfold_8x8x240` | space-to-depth, 4 stride-2 streams |
| `layernorm_16x240` | layer norm |
| `residual_add_16x240` | residual add |
| `qk_scores_16x60` | QKᵀ scores, **query-major** |
| `attn_v_16x60` | attn@V, query-major P, uses **AGG** |
| `attn_scores_km_16x60` | QKᵀ scores, **key-major** |
| `attn_v_bcast_60` | attn@V, key-major P, broadcast (**ACC.ADD**, no AGG) |
| `matmul_240x240_x128` | projection (single token group) |
| `matmul_480x240_x128` | projection (single token group) |
| `matmul_720x240_x128` | projection (single token group) |
| `matmul_240x480_x128` | projection (single token group) |

## Not layer-specific

| Kernel | Why |
|---|---|
| `fully_connected` | FC block, not part of a transformer layer |
| `layernorm_128x16` | 16-channel layer norm, not one of the three layer shapes |
| `matmul_64x64x64` | generic matmul harness |
| `matmul_128x64x64` | generic matmul harness |
| `matmul_128x64x128` | generic matmul harness |
| `matmul_128x128` | generic matmul harness |

---

## Two attention mappings — never mix them

Each layer implements **both** score mappings, and the chains are used end to end
without reshaping in between:

- **query-major:** `qk_scores_*` → `attn_v_*`
- **key-major:** `attn_scores_km_*` → `attn_v_bcast_*`

They produce bit-different results by design and have separate goldens.

**The `attn_v_bcast_*` kernels use `MULT.RC.VE` + `ACC.ADD[.FIRST]` — no AGG.**
Their references are a per-step-rounded float32 left-fold. The AGG-based
`attn_v_*` siblings instead need a per-128-chunk float64 left-fold rounded to
float32 on each `R_ACC` write. Using the AGG recipe for a bcast kernel (or vice
versa) yields a subtly wrong golden that can still pass a loose tolerance.

## Layer-independent conventions

- One output channel per XMEM row; sub-row store strides are a bug. Where a
  channel's output is shorter than a row, the row is still exclusively its own
  and the harness crops in `teardown()`.
- Memory-region bases are derived from row counts, never hardcoded byte maps
  (a byte map sized for 1-byte elements overflows 4× at FP32).
- Every store goes through `ACTIVATE.QUANTIZE` + `STR_POST_AAQ_REG`, co-issued in
  one VLIW word where the activation is `identity`. The simulation-only
  `STR_ACC_REG` appears in no kernel.
- Each kernel owns its `gen_debug_data.py` and imports only from its own package
  (plus `ipu_apps.base`), so any single kernel can be cherry-picked alone.
- Unfold emits **per-stream** output; the packed `[k][p·n]` layout remains a
  deferred experiment.
