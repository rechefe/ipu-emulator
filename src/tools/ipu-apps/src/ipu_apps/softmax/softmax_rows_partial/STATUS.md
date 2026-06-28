# softmax_rows_partial — status

Packed row-softmax for N<=128, P=128/ps rows per 128-lane chunk.

## Working (verified vs numpy, err ~1e-7)
- P=1  (N 65..128): all chunk counts.
- P=2  (N 33..64):  all chunk counts.
- P=4  (N 17..32):  all chunk counts.
- P=8  (N 1..16):   **1 and 2 chunks only**.

## Known bug
- **P=8 (N<=16) with >=3 chunks**: Pass-4 output is wrong *from the first
  write of chunk 0* (values >1, err ~10, growing with chunk count). Passes 1-3
  verified exact even in the failing case (maxvec, numerators, rvec all match
  numpy). The corruption is isolated to Pass 4 (the packed repack) and depends
  on num_chunks>=3 specifically at P=8 — P=4 multi-chunk is fine. Chunk 0 is not
  corrupted by later chunks; its own Pass-4 computation is wrong. Root cause not
  yet found; suspect an LR/slide interaction unique to the 7-iteration p4_part
  loop combined with the multi-chunk outer loop.

## Verified mechanics (probe-level)
- Per-partition reduction via cyclic-offset slide + masked AGG (valid_elements=N).
- exp2 / reciprocal masked to N within each partition.
- Negative-slide repack into r_acc lanes p*ps, full-width ACTIVATE drain.
- #157 fused loads; read-only CR0/CR1; CR8=128 maxvec-select base.

## Next step
Trace the P=8 >=3-chunk Pass-4 chunk-0 computation cycle-by-cycle to find why a
larger padded_rows/num_chunks changes chunk-0's result. Likely a register reused
across the chunk loop and partition loop that isn't reset, or an LR value that
only goes out of range when the loops are deep enough.
