# softmax: five FP32 wide-vector kernels

### Description

The emulator has no softmax. This adds the family: five kernels covering every
1-D softmax shape, split across two axes — which direction the reduction runs,
and how the reduced length relates to the 128-lane datapath.

| kernel | handles |
|---|---|
| `softmax_rows` | rows of exactly 128 elements |
| `softmax_rows_partial` | rows < 128, several packed per 128-lane chunk |
| `softmax_rows_long` | rows > 128, reduced across chunks |
| `softmax_columns` | down each column, width ≥ 65 |
| `softmax_columns_packed` | down each column, width ≤ 64 |

All five share one reformulation, base 2:

```
softmax(x_i) = 2^(c*(x_i - xmax)) / SUM_j 2^(c*(x_j - xmax)),  c = log2(e)
```

so the IPU's native `exp2` activation applies directly, with `c = log2(e)`
resident in a 128-lane vector. All five run the same four passes — row max,
numerators, sum, normalise — using `ACC.SUB` against CR1 for the max subtraction.

### Two properties that hold across the family

Both are load-bearing, and both are pinned by tests rather than asserted in
comments.

**1. No row-count cap.** Per-row scalars (`maxvec`/`rvec`) hold one slot per row
in a single 128-lane vector, so at most 128 rows can be in flight. Each kernel
therefore processes rows in GROUPS of at most 128, running all four passes per
group before advancing.

This is a correctness requirement, not loop bookkeeping. The row index feeds
`MULT.RC.VE`'s `src` scalar-select and `AGG`'s destination slot, both of which
index R0 for 0..127 and switch to the never-loaded R1 at 128. Restarting the
index each group makes that overflow structurally unreachable rather than
merely avoided. Group size is exact — `min(128, rows_left)` — so no padding
chunk is ever processed.

The LR slot has no multiply, which shapes the implementation: where a product
is needed (the next group's base; the group's row count in a chunk-counted
pass) it is taken from an already-walked pointer rather than recomputed.

**2. Output file layout equals input file layout.** Internal layouts differ
considerably — `softmax_rows_partial` packs P logical rows per chunk and pads
the row count up to a multiple of P; `softmax_rows_long` pads a tail chunk —
but every app reads and writes row-major `(rows, cols)` float32. Callers never
un-pack. Only the intermediate NUM region stays unpacked on-device, which is
what keeps the per-row reduce cheap.

### Implementation notes

`softmax_rows_partial/STATUS.md` records the two bugs that cost real debugging
time and the reasoning behind their fixes:

- **Pass-4 cross-partition contamination.** `ACC.ADD`/`ACC.ADD.FIRST` write all
  128 lanes unconditionally, while `MULT.RC.VE` leaves other rows' real data in
  the out-of-range lanes, so partitions corrupted each other. Fixed with a
  per-partition `R_MASK` slot.
- **Why that masking needed a Jinja unroll rather than a runtime loop.**
  `MULT.RC.VE`'s `mask_offset` is a compile-time immediate, so a single loop
  body cannot vary it per partition, and `mask_shift` shifts by too little to
  span a partition boundary.

### Acceptance

- [ ] All five kernels match a numpy reference to ~1e-7 across their domain
- [ ] Shapes well past 128 rows verified (n=128 rows=300, n=64 rows=1000,
      n=16 rows=500, n=8 rows=1032, n=20 rows=2000)
- [ ] Row/column sums exactly 1.0
- [ ] Output file layout equals input file layout for every kernel
- [ ] Bazel targets added for all five test files
- [ ] Full test suite green
