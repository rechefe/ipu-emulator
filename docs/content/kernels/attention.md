# Attention

Twelve kernels implementing self-attention's two matmuls (`QKᵀ` and `attn·V`)
across MobileViT-S's three transformer layers. Like every other kernel in this
section, each is a **wide-vector FP32**, exact-shape match.

## Two mappings — never mix them

There are **two independent, non-interchangeable** ways to compute the same
attention scores and output, differing in which operand owns a whole XMEM
row:

- **query-major**: `qk_scores` → `attn_v` (scores reduced via `AGG`,
  attn·V summed via `AGG.SUM.FIRST`'s float64 left-fold)
- **key-major**: `attn_scores_km` → `attn_v_bcast` (scores stored key-major,
  attn·V summed via a single continuous float32 `ACC.ADD` fold, no `AGG`)

Both compute the same mathematical result but round differently along the
way, so they produce **bit-different output**. A chain must stay entirely on
one mapping — feeding `qk_scores`'s output into `attn_v_bcast`, or vice versa,
silently produces wrong numbers, not an error.

This is why the two mappings are four **separate operations** in the
registry (`qk_scores`, `attn_scores_km`, `attn_v`, `attn_v_bcast`), not four
variants of one `op="attention"` — the registry cannot accidentally pick
one chain's kernel to complete the other's query.

## `qk_scores` — query-major QKᵀ

```
S[i, s] = sum_c Q[c, i] * K[c, s]
```

stored **query-major**: one whole row per query token. Every head is
processed in one call; `qk_scores_64x48`/`qk_scores_256x36` additionally
cover all 4 pixel-streams per call, `qk_scores_16x60` covers one
(stream, head) pair per call.

| Kernel | n_tok | d (head_dim) |
|---|---|---|
| `qk_scores_16x60` | 16 | 60 |
| `qk_scores_64x48` | 64 | 48 |
| `qk_scores_256x36` | 256 | 36 |

## `attn_scores_km` — key-major QKᵀ, one selected head

Same mathematical scores as `qk_scores`, stored **key-major**: one whole row
per key token, so a key's whole score column is contiguous — what the
downstream broadcast attn·V needs. Takes a `head` kwarg (range-checked
`0 <= head < 4`) selecting which of the input file's 4 heads to score; every
head is equally supported, `head` does not affect routing.

| Kernel | n_tok | d (head_dim) |
|---|---|---|
| `attn_scores_km_16x60` | 16 | 60 |
| `attn_scores_km_64x48` | 64 | 48 |
| `attn_scores_km_256x36` | 256 | 36 |

## `attn_v` — query-major attn·V (AGG)

```
O[h, i, t] = sum_s P[h, i, s] * V[h, t, s]
```

`P` staged query-major (one row per query, all keys), `V` staged channel-major
(one row per (head, channel)). Reduces via `AGG.SUM.FIRST`: float32 lane
products, left-folded starting from a Python float and rounded once on the
`R_ACC` write — not the same computation as a plain float32 dot product.

| Kernel | n_tok | d (head_dim) |
|---|---|---|
| `attn_v_16x60` | 16 | 60 |
| `attn_v_64x48` | 64 | 48 |
| `attn_v_256x36` | 256 | 36 |

## `attn_v_bcast` — key-major attn·V (broadcast ACC.ADD)

Same output as `attn_v`, computed via a single continuous per-lane float32
`ACC.ADD` fold (`ACC.ADD.FIRST` at the first key) instead of `AGG` — no
cross-chunk reset, no float64 intermediate. `P` staged key-major (one row per
key, all queries) — the transpose of `attn_v`'s `P` layout, despite sharing
the same `p_path`/`v_path` constructor kwargs.

Indexed by `d` alone: `n_tok` is a fixed module constant per app, not a
caller-visible parameter (there is nothing for a query to assert it
against).

| Kernel | d (head_dim) | n_tok (fixed) |
|---|---|---|
| `attn_v_bcast_36` | 36 | 256 |
| `attn_v_bcast_48` | 48 | 64 |
| `attn_v_bcast_60` | 60 | 16 |

## Picking one

```python
from ipu_apps.kernel_registry import resolve

resolve("qk_scores", n_tok=64, d=48)
resolve("attn_scores_km", n_tok=64, d=48)
resolve("attn_v", n_tok=64, d=48)
resolve("attn_v_bcast", d=48)
```

## Using one directly

```python
from ipu_apps.kernel_registry import resolve

verdict = resolve("qk_scores", n_tok=64, d=48)
app = verdict.kernel.app_class(
    inst_path="qk_scores_64x48.bin",
    query_path="q.bin",
    key_path="k.bin",
    output_path="scores.bin",
)
state, cycles = app.run()
```
