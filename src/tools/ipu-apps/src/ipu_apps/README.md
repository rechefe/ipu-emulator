# MobileViT kernel library

This is the full L3/L4/L5 MobileViT kernel library — Bazel-registered
production `.asm` kernels, one directory per kernel
(`<kernel_name>/<kernel_name>.asm`), each with a companion `__init__.py`
harness. Every kernel's header comment carries a `Layer`/`Scope`/`Layout`/
`Shape`/`Status`/`Related`/`Tests` field block; this file is a directory-wide
index over those fields, grouped by algorithmic family.

For the "packed activation layout" experiment (an alternative XMEM layout
for L4/L5 built on top of several of these kernels), see the separate,
untracked, standalone kernel set and index at
[`src/tools/ipu-apps/test/README.md`](../test/README.md).

For deep dives beyond what's summarized here: `kernel_docs/L3_kernel_reference.md`
(11 L3 kernels, full assembly transcription + measured perf + per-bundle
walkthroughs) and `kernel_docs/kernel_layer_map.md` (per-layer role tables,
attention mapping conventions, cross-lane reduction rules, the packed-layout
design).

## Index

Kernels ported across all three MobileViT layers (L3/L4/L5) are listed as
one row spanning all three shapes rather than three separate rows.

| Kernel family | L3 shape | L4 shape | L5 shape | Scope | Role |
|---|---|---|---|---|---|
| `attn_scores_km_*` | 256x36 | 64x48 | 16x60 | single-stream | Key-major attention scores |
| `attn_v_*` | 256x36 | 64x48 | 16x60 | single-stream | Query-major attn@V (AGG) |
| `attn_v_bcast_*` | 36 | 48 | 60 | single-stream | Key-major attn@V (no AGG) |
| `qk_scores_*` | 256x36 | 64x48 | 16x60 | single-stream | Query-major QKᵀ scores |
| `layernorm_*` | 256x144 | 64x192 | 16x240 | single-stream | LayerNorm |
| `residual_add_*` | 256x144 | 64x192 | 16x240 | single-stream | Elementwise residual add |
| `unfold_*` | 32x32x144 | 16x16x192 | 8x8x240 | single-stream | Patch extraction (spatial → token/channel) |
| `matmul_*_x128` (OutProj) | 144x144 | 192x192 | 240x240 | single-stream | Core matmul: output projection |
| `matmul_*_x128` (FFN1) | 288x144 | 384x192 | 480x240 | single-stream | Core matmul: FFN expansion (silu) |
| `matmul_*_x128` (FFN2) | 144x288 | 192x384 | 240x480 | single-stream | Core matmul: FFN contraction |
| `matmul_*_x128` (QKV) | 432x144 | 576x192 | 720x240 | single-stream | Core matmul: fused Q+K+V projection |
| `proj_outproj_*_p4` | 144 | 192 | 240 | all-stream/P4 | P4 output projection |
| `proj_ffn1_*_p4` | 144 | 192 | 240 | all-stream/P4 | P4 FFN expansion (silu) |
| `proj_ffn2_*_p4` | 144 | 192 | 240 | all-stream/P4 | P4 FFN contraction |
| `proj_qkv_*_p4` | 144 | 192 | 240 | all-stream/P4 | P4 fused Q+K+V projection |
| `matmul_128x128`, `matmul_128x64x128`, `matmul_128x64x64`, `matmul_64x64x64` | shared/multi-layer | | | single-stream | Generic matmul harness, not tied to one layer |
| `fully_connected` | n/a | | | single-stream | Generic FC primitive (128→64, 10 samples) |

The `144`/`192`/`240` channel-count suffix on `matmul_*_x128` and `proj_*_p4`
kernels maps directly to L3/L4/L5 respectively.

## By family

### Attention

`qk_scores_*` (query-major QKᵀ) and `attn_scores_km_*` (key-major QKᵀ, same
math, transposed output layout) are siblings feeding two different
downstream attn@V constructions: `attn_v_*` consumes `qk_scores_*`'s
query-major output via `AGG`-based reduction; `attn_v_bcast_*` consumes
`attn_scores_km_*`'s key-major output via a broadcast construction with no
`AGG`. All four families are ported across L3 (`_256x36`/`_36`) → L4
(`_64x48`/`_48`) → L5 (`_16x60`/`_60`), with L4 additionally streaming all 4
P streams x 4 heads per invocation.

### LayerNorm / residual add

`layernorm_*` and `residual_add_*` are both ported L3→L4→L5 with the L3
shape as the original and L4/L5 as direct ports (`layernorm_128x16` is a
separate 16-channel ancestor kernel, not one of the three layer shapes).
`residual_add_*` consumes each layer's `layernorm_*` output directly.

### Unfold

`unfold_32x32x144` (L3) is the original; `unfold_16x16x192` (L4) and
`unfold_8x8x240` (L5) **re-derive** the spatial geometry rather than being
direct ports (stripe count and `elements_in_row` both change per layer).
Each emits 4 spatial streams and feeds that layer's `layernorm_*`.

### Core matmul (`matmul_*_x128`)

The shared building block underneath every projection: 4 near-identical
kernels per layer (OutProj, FFN1, FFN2, fused QKV), differing only in K/N
and activation (FFN1 uses silu, the rest identity). Plus 4
layer-independent generic variants (`matmul_128x128`, `matmul_128x64x128`,
`matmul_128x64x64`, `matmul_64x64x64`) not tied to any MobileViT shape — see
`kernel_docs/kernel_layer_map.md`'s "Not layer-specific" table.

### P4 projections (`proj_*_p4`)

All-stream (4-partition) counterparts of the OutProj/FFN1/FFN2/QKV matmul
family, one set per layer. These are the most heavily documented headers in
the library (rationale sections like "WHY STREAM OUTERMOST").

### Fully connected

`fully_connected` is a standalone generic FC primitive (128→64, 10 samples),
not tied to a specific MobileViT layer — the basis `matmul_128x64x128`'s
streaming shape was built from.

## What's not here

`rc_idx`/loop-bound formulas, per-bundle cycle annotations, measured
performance numbers, and the deep per-kernel design narrative all live in
`kernel_docs/L3_kernel_reference.md` (11 L3 kernels) and
`kernel_docs/kernel_layer_map.md` (all layers, including the "Not
layer-specific" and packed-activation-layout sections) — this README
intentionally doesn't reproduce them.
