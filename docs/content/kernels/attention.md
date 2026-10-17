# Attention

## Two mappings

Two ways to compute the same attention, differing in which operand owns a
whole XMEM row:

- **query-major**: `qk_scores` → `attn_v` (scores reduced via `AGG`,
  attn·V summed via `AGG.SUM.FIRST`'s float64 left-fold)
- **key-major**: `attn_scores_km` → `attn_v_bcast` (scores stored key-major,
  attn·V summed via a single continuous float32 `ACC.ADD` fold, no `AGG`)

The two round differently (bit-different output). Feeding one chain's scores
into the other chain's attn·V silently produces wrong numbers, so each stage
of each chain is its own op, and the registry never hands one chain's scores
to the other's consumer.

All twelve kernels are folders under `kernels/attention/`, run and tested as
`//kernels/attention/<kernel>`. Each is held to a matmul reference and, bit
for bit, to its datapath mirror (`ipu_kernels.reference.datapath`: `acc_fold`
for the score kernels and `attn_v_bcast`, `agg_fold` for `attn_v`;
`kernels/attention/test_datapath.py`). `kernels/attention/test_seams.py` runs
each chain on one `Chain`: every head's score step writes its rows straight
into that head's block of the attn·V kernel's `P`, which reads them in place.

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
per key token. Each call scores one head: its inputs are that head's Q and K.
The `head` parameter (0–3, default 0) names which of the 4 a case generates.

| Kernel | n_tok | d (head_dim) |
|---|---|---|
| `attn_scores_km_16x60` | 16 | 60 |
| `attn_scores_km_64x48` | 64 | 48 |
| `attn_scores_km_256x36` | 256 | 36 |

## `attn_v`

```
O[h, t, i] = sum_s P[h, i, s] * V[h, t, s]
```

`P` (input `b`) staged query-major (one row per query, all keys), `V` (input
`a`) staged channel-major (one row per (head, channel)); `O` (output `y`) is
channel-major, one row per (head, channel). Reduces via `AGG.SUM.FIRST`, and
at n_tok = 256 adds the second 128-key chunk with `AGG.SUM`.

| Kernel | n_tok | d (head_dim) |
|---|---|---|
| `attn_v_16x60` | 16 | 60 |
| `attn_v_64x48` | 64 | 48 |
| `attn_v_256x36` | 256 | 36 |

## `attn_v_bcast`

Same output as `attn_v`, summed with per-lane `ACC.ADD` instead of `AGG`.
`P` (input `b`) is staged key-major (one row per key, all queries) — the
transpose of `attn_v`'s `P` layout, under the same name. Query by `d`,
optionally with `n_tok`.

| Kernel | d (head_dim) | n_tok (fixed) |
|---|---|---|
| `attn_v_bcast_36` | 36 | 256 |
| `attn_v_bcast_48` | 48 | 64 |
| `attn_v_bcast_60` | 60 | 16 |

## Queries

```bash
bazel run //kernels:query -- qk_scores n_tok=64 d=48
bazel run //kernels:query -- attn_scores_km n_tok=64 d=48 head=2
bazel run //kernels:query -- attn_v n_tok=64 d=48
bazel run //kernels:query -- attn_v_bcast d=48
```
