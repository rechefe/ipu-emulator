# MobileViT-S

Kernels per transformer layer.

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
| `matmul_144x144_x128` | OutProj, single stream (K=144→144), **identity** |
| `matmul_288x144_x128` | FFN1, single stream (K=144→288), **silu** |
| `matmul_432x144_x128` | QKV, single stream (K=144→432), **identity** |
| `matmul_144x288_x128` | FFN2, single stream (K=288→144), **identity** |
| `proj_qkv_144_p4` | QKV, all P=4 streams in one invocation (K=144→432), **identity** |
| `proj_outproj_144_p4` | OutProj, all P=4 streams in one invocation (K=144→144), **identity** |
| `proj_ffn1_144_p4` | FFN1, all P=4 streams in one invocation (K=144→288), **silu** |
| `proj_ffn2_144_p4` | FFN2, all P=4 streams in one invocation (K=288→144), **identity** |
| `fold_32x32x144` | depth-to-space, inverse of `unfold_32x32x144` |
| `concat_32x32x96x96` | channel concat (residual + fold output) |

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
| `matmul_192x192_x128` | OutProj, single stream (K=192→192), **identity** |
| `matmul_384x192_x128` | FFN1, single stream (K=192→384), **silu** |
| `matmul_576x192_x128` | QKV, single stream (K=192→576), **identity** |
| `matmul_192x384_x128` | FFN2, single stream (K=384→192), **identity** |
| `proj_qkv_192_p4` | QKV, all P=4 streams in one invocation (K=192→576), **identity** |
| `proj_outproj_192_p4` | OutProj, all P=4 streams in one invocation (K=192→192), **identity** |
| `proj_ffn1_192_p4` | FFN1, all P=4 streams in one invocation (K=192→384), **silu** |
| `proj_ffn2_192_p4` | FFN2, all P=4 streams in one invocation (K=384→192), **identity** |
| `fold_16x16x192` | depth-to-space, inverse of `unfold_16x16x192` |
| `concat_16x16x128x128` | channel concat (residual + fold output) |

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
| `matmul_240x240_x128` | OutProj, single stream (K=240→240), **identity** |
| `matmul_480x240_x128` | FFN1, single stream (K=240→480), **silu** |
| `matmul_720x240_x128` | QKV, single stream (K=240→720), **identity** |
| `matmul_240x480_x128` | FFN2, single stream (K=480→240), **identity** |
| `proj_qkv_240_p4` | QKV, all P=4 streams in one invocation (K=240→720), **identity** |
| `proj_outproj_240_p4` | OutProj, all P=4 streams in one invocation (K=240→240), **identity** |
| `proj_ffn1_240_p4` | FFN1, all P=4 streams in one invocation (K=240→480), **silu** |
| `proj_ffn2_240_p4` | FFN2, all P=4 streams in one invocation (K=480→240), **identity** |
| `fold_8x8x240` | depth-to-space, inverse of `unfold_8x8x240` |
| `concat_8x8x160x160` | channel concat (residual + fold output) |

## Chaining rules

- **Two attention mappings, never mixed:** query-major `qk_scores_*` → `attn_v_*`,
  and key-major `attn_scores_km_*` → `attn_v_bcast_*`. Each attn@V kernel only
  reads its own chain's score layout.
- **Q must be pre-scaled.** No score kernel applies `1/√head_dim`; fold it into
  the Q rows of the QKV weights once:

    | Layer | head_dim | scale |
    |---|---|---|
    | L3 | 36 | 0.166667 |
    | L4 | 48 | 0.144338 |
    | L5 | 60 | 0.129099 |

- **Producers write full rows; only the final consumer crops.** One output
  channel per XMEM row, even where fewer than 128 lanes are valid.
