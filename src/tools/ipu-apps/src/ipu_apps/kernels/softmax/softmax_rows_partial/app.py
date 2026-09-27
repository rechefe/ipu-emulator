"""Row-softmax for N < 128 elements/row, packed P rows per 128-element chunk.

Logical rows of ``N`` elements are packed into 128-element physical chunks. The
partition size ``ps`` is the next power of two >= N (clamped to [16, 128]), so
``P = 128 / ps`` rows share one chunk:

    N in 65..128 -> ps=128, P=1     N in 17..32 -> ps=32,  P=4
    N in 33..64  -> ps=64,  P=2     N in  1..16 -> ps=16,  P=8

Row p occupies elements ``p*ps .. p*ps + N-1`` of its chunk; elements ``N..ps-1`` are
padding. ``CR15.valid_elements = N`` masks every AGG/ACTIVATE so only the first
N elements of each partition contribute (max/sum ignore the padding tail).

Reduction trick (probe-validated): to reduce partition p, ``MULT.RC.VV`` reads
r_cyclic at ELEMENT offset ``p*ps`` (rc_idx is element-addressed, matching
``LDR_CYCLIC_MULT_REG``'s index -- issue #182/PR #196) so partition p lands in
mult_res elements 0..N-1, then a masked ``AGG`` reduces exactly those into r_acc
slot = row index.

Layout:
  * input / output : row-major (rows x N) in the FILE, identical on both sides.
                     On-device both are PACKED (P rows/chunk); setup adds
                     the partition/row padding on the way in and teardown drops
                     it on the way out, so the output file has exactly the input
                     file's shape.
  * numerators     : UNPACKED (one 512B chunk per logical row, elements 0..N-1) --
                     intermediate, free to be convenient.
  * maxvec/rvec    : 128-element scalar vectors, slot per logical row.

Only Pass 4 re-packs: for partition p, ``MULT.RC.VE`` places row p's product
at elements ``[p*ps, p*ps+N)`` (same rc_idx trick, reversed), masked via a
per-partition ``R_MASK`` slot (``mask_offset=p``, built by ``_partition_masks``
and loaded once per chunk via ``LDR_MULT_MASK_REG``) so every OTHER element is
zeroed before ``acc.add``/``acc.add.first`` accumulates it into the shared
r_acc -- this is what lets P separate partitions safely share one accumulator
without corrupting each other (see STATUS.md's "Pass-4 cross-partition
contamination" for why this masking is necessary). One ACTIVATE+store then
drains the fully assembled packed chunk. ``mask_offset`` is a compile-time
immediate, so the per-partition instruction sequence is unrolled per P value
via Jinja in the ``.asm`` (one block each for P=1/2/4/8), with a runtime
dispatch on P selecting which block runs.

The base softmax_rows app is the ps=128/P=1 special case; this app handles the
P>1 packed regimes. Same base-2 / max-subtraction math (see softmax_rows).

Usage::

    from ipu_apps.kernels.softmax.softmax_rows_partial.app import SoftmaxRowsPartialApp
    app = SoftmaxRowsPartialApp(
        inst_path="softmax_rows_partial.bin",
        input_path="logits.bin",   # rows * N float32, row-major
        output_path="probs.bin",
        n=32, rows=100,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from ipu_apps.kernels.softmax.app import LANES, SoftmaxApp, partition_size, softmax_spec

# r_cyclic's full ring: 512 ELEMENTS (MULT.RC's rc_idx is element-addressed,
# matching LDR_CYCLIC_MULT_REG's index -- issue #182/PR #196). The Pass-4
# repack's reverse slide wraps at this element count.
RING_ELEMENTS = 512


class SoftmaxRowsPartialApp(SoftmaxApp):
    """Packed row-softmax for ``n <= 128``: ``P = 128 / ps`` rows per chunk.

    Rows are padded up to a multiple of P on device; the output file drops the
    padding again. maxvec/rvec hold one slot per logical row, so the .asm runs
    groups of at most 128 logical rows (``128 / P`` chunks) -- a correctness
    bound: a row index >= 128 would select the unloaded R1.
    """

    def _geometry(self):
        self.ps = partition_size(self.n)
        self.parts_per_chunk = LANES // self.ps
        return self.ps

    @property
    def valid_elements(self):
        return self.n  # every AGG/ACTIVATE sees only the first N elements

    def _layout_resident(self):
        super()._layout_resident()
        # Pass-4 R_MASK image; the .asm derives its row as rvec row + 1.
        self.mask_addr = self._alloc()

    def _write_resident(self, state):
        state.xmem.write_address(self.mask_addr, self._partition_masks())

    def _partition_masks(self) -> bytes:
        """128-byte R_MASK image: 8 slots of 16 bytes (128 bits).

        Slot p (p < P) keeps bits [p*ps, p*ps+N) and zeroes every other bit
        (bit 1 = keep). Each partition's MULT.RC.VE (mask_offset=p) is thereby
        restricted to exactly its own elements, so acc.add never leaks another
        row's data into a neighbouring partition (see STATUS.md).
        """
        buf = bytearray(128)
        for p in range(self.parts_per_chunk):
            bits = ((1 << self.n) - 1) << (p * self.ps)
            buf[p * 16:(p + 1) * 16] = bits.to_bytes(16, byteorder="little")
        return bytes(buf)

    def _crs(self):
        return {
            8: LANES,                              # R1 byte-index base for maxvec select
            9: self.device_rows,                   # total chunks (group loop bound)
            11: RING_ELEMENTS,                     # r_cyclic ring size (elements)
            12: self.ps,                           # partition element stride
            13: LANES // self.parts_per_chunk,     # chunks in a full group
            14: self.parts_per_chunk,              # P
        }


def _explain(q):
    ps = partition_size(q.n)
    p = LANES // ps
    return (f"n ({q.n}) < {LANES}: packed row kernel, partition size ps={ps}, "
            f"P={p} logical rows per {LANES}-element chunk, in groups of "
            f"{LANES // p} chunks.")


SPEC = softmax_spec(
    SoftmaxRowsPartialApp, along_rows=True,
    supports=lambda q: (None if q.n <= LANES else
                        f"packs whole rows into one {LANES}-element chunk, so it needs a row "
                        f"of at most {LANES} elements; this row has {q.n}"),
    explain=_explain,
    # This kernel also handles n == 128 (P=1), as does softmax_rows. `supports`
    # says what it CAN do; `cost` makes the specialised full-width kernel win
    # at exactly 128 while this one stays the choice below it.
    cost=lambda q: 2.0 if q.n == LANES else 1.0,
    tags=("packed",),
)
