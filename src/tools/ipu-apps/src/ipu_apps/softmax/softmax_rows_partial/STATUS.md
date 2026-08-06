# softmax_rows_partial — status

Packed row-softmax for N<=128, P=128/ps rows per 128-element chunk.

## Working (verified vs numpy, err ~1e-7)
- P=1 (N 65..128), P=2 (N 33..64), P=4 (N 17..32), P=8 (N 1..16): all chunk
  counts, any number of rows (the 127-row cap is fixed -- see below). Via a
  per-partition R_MASK approach (see below) -- this app was previously
  broken for P>1 past a data-dependent chunk-count threshold (P=8 >=3
  chunks was the only originally-documented case, but P=2 and P=4 had the
  same latent bug, just not yet triggered by the existing test suite).

## Fixed bug: Pass-4 cross-partition contamination

Pass 4 re-packs P logical rows into one chunk. For partition p, `acc.add`
(or `acc.add.first` for p=0) accumulates `MULT.RC.VE`'s result into the
shared `r_acc`. `MULT.RC.VE` always produces 128 real elements -- only
`[p*ps, p*ps+N)` is *this* partition's actual data; the rest is real (not
zero) data belonging to OTHER rows, picked up via the `rc_idx`
cyclic-wraparound placement trick. Without masking, `acc.add`/
`acc.add.first` write/accumulate all 128 elements unconditionally, so each
partition's out-of-range elements got baked into `r_acc` and corrupted the
next partition's write region.

This was invisible at low chunk counts because the leaked cross-row values
happened to be numerically small (well under the 1e-4 test tolerance) --
it was never actually correct for P>1, just lucky. A benchmark sweep with
larger N/rows configurations surfaced it across P=2 and P=4 too, not just
the originally-documented P=8 corner.

Passes 1-3 (maxvec, NUM, rvec) were never affected -- `AGG.MAX.FIRST`/
`AGG.SUM.FIRST` (Passes 1/3) correctly mask via `valid_elements`, while
`ACC.ADD`/`ACC.ADD.FIRST` (Pass 4) don't mask at all on their own.

### Why R_MASK-based masking needed a Jinja unroll, not a runtime loop
`MULT.RC.VE`'s `mask_offset` operand is a compile-time immediate (baked
into the instruction at assemble time), not a runtime register. Since
`mask_offset` must equal the partition index `p` for the mask to isolate
that partition's own elements, and `p` ranges over `0..P-1` where P is only
known at *runtime* (from N), a single runtime loop body can't vary
`mask_offset` per iteration. `mask_shift` IS a runtime register, but only
shifts by up to +/-3 bits -- far too little to span a 16..128-element
partition boundary, so it can't substitute either.

### The fix
1. **Harness** (`_partition_masks()` in `__init__.py`): builds a 128-byte,
   8-slot R_MASK image -- slot p (`p` in `0..P-1`) has bits
   `[p*ps, p*ps+N)` set (keep) and every other bit clear (zero), matching
   the mask polarity convention (bit 1 = keep, bit 0 = zero). Unused slots
   (`p >= P`) stay all-zero, never selected. Written to `PART_MASK_ADDR`
   (placed the row immediately after RVEC, since no CR slot was free for a
   new base address -- its row number is derived in the `.asm` as
   `cr6 (RVEC row) + 1`).
2. **`.asm`**: `LDR_MULT_MASK_REG` loads the whole 8-slot image once per
   Pass-4 entry. Since P (and thus the number of unrolled iterations) is
   only known at runtime, but `mask_offset` must be a literal per
   instruction, the per-partition body is unrolled via Jinja
   (`p4_partition_block(p, is_last)` macro) into 4 separate blocks --
   `p4_run_p1`, `p4_run_p2`, `p4_run_p4`, `p4_run_p8` -- each calling
   `MULT.RC.VE ... mask_offset=p ...` with a literal `p` for every
   partition 0..P-1. A runtime dispatch on `cr14` (P) at the top of
   `p4_chunk` picks which block to enter; each block ends with a branch to
   the shared `p4_drain`.
3. With each partition's `MULT.RC.VE` now masked to exactly its own element
   range (every other element -> `PadMode.ZERO`), `acc.add`/`acc.add.first`
   only ever add zero outside a partition's own slot -- partitions can no
   longer contaminate each other regardless of P.

Verified: all previously-broken configs (P=2 at num_chunks=35, P=4 at
num_chunks=9, P=8 at num_chunks=3/8, N=16 P=8 at num_chunks=5) now match
numpy to ~1e-7. The `NotImplementedError` guard for P=8 >=3 chunks has been
removed from `__init__.py` (no longer needed).

## Fixed bug: row-count overflow (all P)

`lr_row` (the logical-row counter) is used directly as `MULT.RC.VE`'s `src`
operand in several places (e.g. Pass 4's `MULT.RC.VE {{lr_rslide}} {{lr_row}}
...`), where `src`'s LR value selects a scalar from R0 (0..127) or R1
(128..255). It is also `AGG`'s destination slot. When `lr_row` was a *global*
counter it exceeded 127 once total rows did, silently indexing R1 -- which is
never loaded in these passes -- so results were wrong for any row index >=128
(P=1/N=128 broke at rows=129+; P=2/N=64 at rows=130).

### The fix: a group loop (same shape as `softmax_rows`)
The real constraint is that `maxvec`/`rvec` are single 128-element vectors with
one slot per logical row, so at most 128 logical rows can be in flight. All
four passes now run over a GROUP of at most `128/P` chunks (= at most 128
logical rows), and the whole group is processed Pass 1..4 before the next
begins. `lr_row` restarts at 0 each group, so it is structurally incapable of
reaching 128.

Mechanics:
- `CR9` changed from `padded_rows` to `num_chunks` (total, the group-loop
  bound); `CR13` from `num_chunks` to `chunks_per_group` (= `128/P`).
- `lr_cbound = min(CR13, total - done)` makes the last group short, so no
  padding chunk is ever processed.
- Pass 3 counts rows rather than chunks, so it needs the group's *row* count.
  The LR slot has no multiply, so instead of computing `cbound*P`, Pass 1's
  `lr_row` (which steps once per partition per chunk) is captured at the end
  of the pass -- it already equals exactly that product.
- The next group's base is likewise taken from the walked pointer
  (`lr_coff` after Pass 4) rather than recomputed.
- Input/output regions are now sized per instance in `_layout()`: the old
  fixed 0x10000/0x30000 slots held only 128 chunks each, which the group loop
  can now exceed. NUM keeps its fixed address -- it only ever spans one group.

Verified vs numpy across P=1/2/4/8 well past the old cliff (e.g. N=128
rows=300, N=64 rows=1000, N=16 rows=500, N=8 rows=1000, N=20 rows=2000), all
~1e-7, plus the whole pre-existing test suite. The same group loop was added
to `softmax_rows_long` (which had the identical single-group limit).

## Verified mechanics (probe-level)
- Per-partition reduction via cyclic-offset slide + masked AGG (valid_elements=N).
- exp2 / reciprocal masked to N within each partition.
- Negative-slide repack into r_acc elements p*ps, full-width ACTIVATE drain,
  now correctly masked per-partition (see fix above).
- #157 fused loads; read-only CR0/CR1; CR8=128 maxvec-select base.
- rc_idx (MULT.RC.* operand) is ELEMENT-indexed, matching LDR_CYCLIC_MULT_REG's
  index (issue #182/PR #196) -- confirmed correct in this app's CR11/CR12
  (RING_ELEMENTS=512, partition stride=ps, not the old byte-scaled constants).

## Next steps
No known correctness bugs. Possible future work:
1. Performance: the group loop re-loads C_VEC and re-runs the Pass-1..4
   prologue per group; negligible for large groups, but it is per-128-rows
   overhead that could be hoisted.
2. The on-device packing (P logical rows per chunk, rows padded to a multiple
   of P) is invisible in the files -- `teardown` un-packs to row-major, so the
   output file has the same layout as the input. Only the intermediate NUM
   region stays unpacked, which is what keeps the per-row reduce cheap.
