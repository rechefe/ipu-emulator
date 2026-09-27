# Attention

## Two mappings

Two ways to compute the same attention, differing in which operand owns a
whole XMEM row:

- **query-major**: `qk_scores` → `attn_v` (scores reduced via `AGG`,
  attn·V summed via `AGG.SUM.FIRST`'s float64 left-fold)
- **key-major**: `attn_scores_km` → `attn_v_bcast` (scores stored key-major,
  attn·V summed via a single continuous float32 `ACC.ADD` fold, no `AGG`)

The two round differently (bit-different output). Feeding one chain's scores
into the other chain's attn·V silently produces wrong numbers.

## `qk_scores`

```
S[i, s] = sum_c Q[c, i] * K[c, s]
```

stored **query-major**: one whole row per query token. `qk_scores_64x48`
covers all 16 (stream, head) pairs per call; `qk_scores_256x36` and
`qk_scores_16x60` score one head per call.

| Kernel | n_tok | d (head_dim) |
|---|---|---|
| `qk_scores_16x60` | 16 | 60 |
| `qk_scores_64x48` | 64 | 48 |
| `qk_scores_256x36` | 256 | 36 |

## `attn_scores_km`

Same mathematical scores as `qk_scores`, stored **key-major**: one whole row
per key token. Optional query parameter `head` (0–3, default 0) selects which
of the input file's 4 heads to score.

| Kernel | n_tok | d (head_dim) |
|---|---|---|
| `attn_scores_km_16x60` | 16 | 60 |
| `attn_scores_km_64x48` | 64 | 48 |
| `attn_scores_km_256x36` | 256 | 36 |

## `attn_v`

```
O[h, i, t] = sum_s P[h, i, s] * V[h, t, s]
```

`P` staged query-major (one row per query, all keys), `V` staged channel-major
(one row per (head, channel)). Reduces via `AGG.SUM.FIRST`.

| Kernel | n_tok | d (head_dim) |
|---|---|---|
| `attn_v_16x60` | 16 | 60 |
| `attn_v_64x48` | 64 | 48 |
| `attn_v_256x36` | 256 | 36 |

## `attn_v_bcast`

Same output as `attn_v`, summed with per-lane `ACC.ADD` instead of `AGG`.
`P` is staged key-major (one row per key, all queries) — the transpose of
`attn_v`'s `P` layout, under the same `p_path` name. Query by `d`, optionally
with `n_tok`.

| Kernel | d (head_dim) | n_tok (fixed) |
|---|---|---|
| `attn_v_bcast_36` | 36 | 256 |
| `attn_v_bcast_48` | 48 | 64 |
| `attn_v_bcast_60` | 60 | 16 |

## Queries

```bash
bazel run //src/tools/ipu-apps:query -- qk_scores n_tok=64 d=48
bazel run //src/tools/ipu-apps:query -- attn_scores_km n_tok=64 d=48 head=2
bazel run //src/tools/ipu-apps:query -- attn_v n_tok=64 d=48
bazel run //src/tools/ipu-apps:query -- attn_v_bcast d=48
```
