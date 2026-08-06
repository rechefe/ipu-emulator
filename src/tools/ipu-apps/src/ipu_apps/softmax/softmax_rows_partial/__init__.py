"""Row-softmax for N < 128 elements/row, packed P rows per 128-lane chunk.

Logical rows of ``N`` elements are packed into 128-lane physical chunks. The
partition size ``ps`` is the next power of two >= N (clamped to [16, 128]), so
``P = 128 / ps`` rows share one chunk:

    N in 65..128 -> ps=128, P=1     N in 17..32 -> ps=32,  P=4
    N in 33..64  -> ps=64,  P=2     N in  1..16 -> ps=16,  P=8

Row p occupies lanes ``p*ps .. p*ps + N-1`` of its chunk; lanes ``N..ps-1`` are
padding. ``CR15.valid_elements = N`` masks every AGG/ACTIVATE so only the first
N lanes of each partition contribute (max/sum ignore the padding tail).

Reduction trick (probe-validated): to reduce partition p, ``MULT.RC.VV`` reads
r_cyclic at ELEMENT offset ``p*ps`` (rc_idx is element-addressed, matching
``LDR_CYCLIC_MULT_REG``'s index -- issue #182/PR #196) so partition p lands in
mult_res lanes 0..N-1, then a masked ``AGG`` reduces exactly those into r_acc
slot = row index.

Layout:
  * input / output : row-major (rows x N) in the FILE, identical on both sides.
                     On-device both are PACKED (P rows/chunk); _pack_input adds
                     the partition/row padding on the way in and teardown drops
                     it on the way out, so the output file has exactly the input
                     file's shape.
  * numerators     : UNPACKED (one 512B chunk per logical row, lanes 0..N-1) --
                     intermediate, free to be convenient.
  * maxvec/rvec    : 128-lane scalar vectors, slot per logical row.

Only Pass 4 re-packs: for partition p, ``MULT.RC.VE`` places row p's product
at lanes ``[p*ps, p*ps+N)`` (same rc_idx trick, reversed), masked via a
per-partition ``R_MASK`` slot (``mask_offset=p``, built by ``_partition_masks``
and loaded once per chunk via ``LDR_MULT_MASK_REG``) so every OTHER lane is
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

    from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp
    app = SoftmaxRowsPartialApp(
        inst_path="softmax_rows_partial.bin",
        input_path="logits.bin",   # rows * N float32, row-major
        output_path="probs.bin",
        n=32, rows=100,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, no, yes
from ipu_apps.softmax._spec_support import (
    WIDE_VECTOR_ONLY,
    positive_dims,
    softmax_query,
)

if TYPE_CHECKING:
    pass

# -- Constants --------------------------------------------------------------

LANES = 128                  # lanes per physical chunk
CHUNK_BYTES = LANES * 4      # 512 bytes per FP32 chunk
# r_cyclic's full ring: 512 ELEMENTS (MULT.RC's rc_idx is element-addressed,
# matching LDR_CYCLIC_MULT_REG's index -- issue #182/PR #196). The Pass-4
# repack's reverse slide wraps at this element count, not CHUNK_BYTES.
RING_ELEMENTS = 512

# XMEM byte-address map. INPUT_BASE non-zero; literal 0 lives in CR0 (read-only).
# INPUT and OUTPUT scale with the total row count, so their bases are computed
# per instance in _layout(); only NUM has a fixed size (one GROUP's worth of
# unpacked rows, i.e. at most LANES chunks, since a group is capped at LANES
# logical rows). The constants below are the small-input defaults / the fixed
# resident vectors.
INPUT_BASE_ADDR = 0x10000    # packed input chunks (default; see _layout)
NUM_BASE_ADDR = 0x20000      # unpacked numerator chunks (one GROUP: <=128)
OUTPUT_BASE_ADDR = 0x30000   # packed output chunks (default; see _layout)
CVEC_ADDR = 0x40000          # C_VEC = log2(e), resident
MAXVEC_ADDR = 0x40400        # per-row max (1 chunk)
RVEC_ADDR = 0x40600          # per-row 1/sum (1 chunk)
# Pass-4 per-partition R_MASK image: 128 B used (8x16B slots), placed in the
# row immediately after RVEC so its row number is derivable as cr6+1 in the
# .asm (no free CR slot remains -- CR0-CR15 are all already assigned).
PART_MASK_ADDR = RVEC_ADDR + CHUNK_BYTES

LOG2E = math.log2(math.e)


def partition_size(n: int) -> int:
    """Next power of two >= n, clamped to [16, 128]."""
    if not 1 <= n <= 128:
        raise ValueError(f"N must be in 1..128; got {n}")
    ps = 16
    while ps < n:
        ps *= 2
    return ps


class SoftmaxRowsPartialApp(IpuApp):
    """Packed row-softmax for N elements/row (N <= 128), P = 128/ps rows/chunk.

    Args:
        inst_path:   Assembled instruction binary.
        input_path:  Input logits, ``rows * N`` float32, row-major (one row's N
                     elements contiguous).
        output_path: Optional output path (packed, same layout as input).
        n:           Elements per logical row (1..128).
        rows:        Number of logical rows (any count; zero-padded up to a
                     multiple of P internally).
    """

    def __init__(self, *, n: int, rows: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.n = int(n)
        self.rows = int(rows)
        # Delegate to the registry declaration rather than restating the bounds:
        # SPEC.supports is the single source of truth for this kernel's domain,
        # so the guard and the router can never disagree.
        SPEC.guard(shape=(self.rows, self.n), dim=1)
        self.ps = partition_size(self.n)
        self.parts_per_chunk = LANES // self.ps          # P
        # Pad logical rows up to a multiple of P so the last chunk is full.
        rem = self.rows % self.parts_per_chunk
        self.padded_rows = self.rows + (self.parts_per_chunk - rem) % self.parts_per_chunk
        self.num_chunks = self.padded_rows // self.parts_per_chunk
        # maxvec/rvec are single 128-lane vectors holding one slot per logical
        # row, so at most LANES logical rows -- i.e. LANES//P chunks -- can be
        # in flight. Rows beyond that are handled by re-running all four passes
        # on the next group (see the .asm group loop). This is a correctness
        # bound, not just a buffer size: a logical-row index >= 128 would
        # overflow MULT.RC.VE's `src` scalar-select into the unloaded R1.
        self.chunks_per_group = LANES // self.parts_per_chunk
        self.rows_per_group = LANES
        self._layout()

    def _layout(self) -> None:
        """Place the two row-count-scaled regions (input, output).

        NUM, C_VEC, maxvec/rvec and the partition-mask image are all fixed size
        (NUM spans one group, the rest one chunk each), so they keep their
        constant addresses. Input and output grow with num_chunks, so they are
        placed above PART_MASK, 64 KiB-aligned and sized to the real chunk
        count -- the old fixed 0x10000/0x30000 slots held only 128 chunks each,
        which the group loop can now exceed.
        """
        region = self.num_chunks * CHUNK_BYTES
        step = max((region + 0xFFFF) & ~0xFFFF, 0x10000)   # 64 KiB align
        base = (PART_MASK_ADDR + CHUNK_BYTES + 0xFFFF) & ~0xFFFF
        self.input_base = base
        self.output_base = base + step

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        state.dtype = DType.INT8
        return state

    def _partition_masks(self) -> bytes:
        """Build the 128-byte R_MASK image for Pass 4's per-partition writes.

        8 slots of 16 bytes (128 bits) each. Slot p (p in 0..P-1) has bits
        [p*ps, p*ps+N) set (keep) and every other bit clear (zero out) --
        see mask polarity convention: bit 1 = KEEP lane, bit 0 = ZERO lane.
        Unused slots (p >= P) are left all-zero (never selected).

        This lets each partition's own MULT.RC.VE (mask_offset=p, mask_shift=0)
        restrict its result to EXACTLY that partition's [p*ps, p*ps+N) lanes --
        every other lane (including the real-but-wrong-row data the rc_idx
        wraparound trick leaves in mult_res) is masked to PadMode.ZERO, so
        acc.add/acc.add.first accumulate 0 there and never contaminate a
        neighboring partition's slot. Fixes the Pass-4 cross-partition
        contamination bug (see STATUS.md) uniformly for every P.
        """
        buf = bytearray(128)
        for p in range(self.parts_per_chunk):
            bits = 0
            for i in range(p * self.ps, p * self.ps + self.n):
                bits |= (1 << i)
            buf[p * 16:(p + 1) * 16] = bits.to_bytes(16, byteorder="little")
        return bytes(buf)

    def _pack_input(self) -> bytes:
        """Read row-major (rows x N) float32, zero-pad to padded_rows, pack into
        chunks of P rows; row p at lanes p*ps..p*ps+N-1, padding lanes zero."""
        raw = self.input_path.read_bytes()
        flat = struct.unpack(f"<{self.rows * self.n}f", raw[: self.rows * self.n * 4])
        packed = bytearray(self.num_chunks * CHUNK_BYTES)
        for r in range(self.rows):  # padded rows stay zero
            chunk = r // self.parts_per_chunk
            p = r % self.parts_per_chunk
            base = chunk * CHUNK_BYTES + p * self.ps * 4
            row = flat[r * self.n:(r + 1) * self.n]
            struct.pack_into(f"<{self.n}f", packed, base, *row)
        return bytes(packed)

    def setup(self, state: "IpuState") -> None:
        state.xmem.write_address(self.input_base, self._pack_input())
        state.xmem.write_address(CVEC_ADDR, struct.pack("<128f", *([LOG2E] * LANES)))
        state.xmem.write_address(PART_MASK_ADDR, self._partition_masks())

        # CR15.valid_elements = N masks every AGG/ACTIVATE to the first N lanes.
        state.set_cr_dstructure(valid_elements=self.n)

        # CR0=0, CR1=1 are read-only hardware constants (reused as zero source /
        # 1.0 identity scalar). Writable CRs below.
        # CR1 == 1 (read-only) serves as the generic +1 increment, freeing a slot.
        # .asm XMEM operands are ROW numbers (one row = CHUNK_BYTES), not byte
        # addresses -- see issue #179 -- so all base CRs below are row numbers.
        state.regfile.set_cr(2, self.output_base // CHUNK_BYTES)
        state.regfile.set_cr(3, CVEC_ADDR // CHUNK_BYTES)
        state.regfile.set_cr(4, NUM_BASE_ADDR // CHUNK_BYTES)
        state.regfile.set_cr(5, MAXVEC_ADDR // CHUNK_BYTES)
        state.regfile.set_cr(6, RVEC_ADDR // CHUNK_BYTES)
        state.regfile.set_cr(7, 1)                    # chunk/row stride: advance one row per chunk
        state.regfile.set_cr(8, LANES)                # 128: R1 byte-index base for maxvec select
        state.regfile.set_cr(9, self.num_chunks)      # total chunks (group loop bound)
        state.regfile.set_cr(10, self.input_base // CHUNK_BYTES)  # input base row
        # CR11 = RING_ELEMENTS (512): r_cyclic ring size for the Pass-4 repack's
        # reverse-slide SUB src_a. MULT.RC's rc_idx is an ELEMENT offset into
        # r_cyclic (matching LDR_CYCLIC_MULT_REG's index -- issue #182/PR #196),
        # not an XMEM address, so it stays element-valued even though CR7
        # switched to rows -- and it must be the FULL ring, not one row's 128
        # elements, since rc_idx wraps at the ring boundary, not the row
        # boundary.
        state.regfile.set_cr(11, RING_ELEMENTS)
        state.regfile.set_cr(12, self.ps)              # partition element stride (ps)
        # CR13 = chunks in a FULL group. maxvec/rvec hold one slot per logical
        # row in a single 128-lane vector, so a group is at most 128 logical
        # rows = 128/P chunks. The .asm caps each group at this and runs a
        # short final group; see the group-loop note in the .asm header.
        state.regfile.set_cr(13, self.chunks_per_group)
        state.regfile.set_cr(14, self.parts_per_chunk)  # P (partitions per chunk)

    def teardown(self, state: "IpuState") -> None:
        """Write the output in the SAME row-major layout as the input file.

        Pass 4 already re-packs on-device: its reverse r_cyclic slide places
        logical row p back at lanes [p*ps, p*ps+N) of the output chunk, so the
        chunk in XMEM mirrors the input chunk exactly. What XMEM additionally
        holds is *padding* the input file never had -- the (ps - N) dead lanes
        inside each partition, and the rows padded up to a multiple of P to
        fill the last chunk. Dropping those here is the read-back mirror of
        _pack_input, and makes the output file byte-for-byte the same shape as
        the input (as it already is for every other softmax app). When N == ps
        and rows % P == 0 there is no padding and this is a straight copy.
        """
        if self.output_path is None:
            return
        raw = state.xmem.read_address(
            self.output_base, self.num_chunks * CHUNK_BYTES
        )
        out = bytearray(self.rows * self.n * 4)
        for r in range(self.rows):          # padded rows are simply not emitted
            chunk = r // self.parts_per_chunk
            p = r % self.parts_per_chunk
            src = chunk * CHUNK_BYTES + p * self.ps * 4
            dst = r * self.n * 4
            out[dst:dst + self.n * 4] = raw[src:src + self.n * 4]
        Path(self.output_path).write_bytes(bytes(out))

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain -- the constructor
# guard delegates to it rather than restating the bounds.


def _supports(**params):
    q = softmax_query(params["shape"], params["dim"])
    bad = positive_dims(q)
    if bad:
        return no(bad)
    if not q.along_rows:
        return no("reduces down columns, not along rows")
    if q.n > LANES:
        return no(
            f"packs whole rows into one {LANES}-lane chunk, so it needs a row "
            f"of at most {LANES} elements; this row has {q.n}"
        )
    return yes()


def _build(**params):
    q = softmax_query(params["shape"], params["dim"])
    return {"n": q.n, "rows": q.rows}


def _explain(**params):
    q = softmax_query(params["shape"], params["dim"])
    ps = partition_size(q.n)
    p = LANES // ps
    return (
        f"n ({q.n}) < {LANES}: packed row kernel, partition size ps={ps}, "
        f"P={p} logical rows per {LANES}-lane chunk, in groups of "
        f"{LANES // p} chunks."
    )


SPEC = KernelSpec(
    name="softmax_rows_partial",
    op="softmax",
    variant="rows_partial",
    app_class=SoftmaxRowsPartialApp,
    asm="softmax_rows_partial.asm",
    tags=("fp32-wide", "packed"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: softmax_query(params["shape"], params["dim"]).bundle,
    # This kernel genuinely handles n == 128 too (that is just P=1), so it and
    # softmax_rows both claim that width. `supports` states what the kernel CAN
    # do; `cost` decides which one should WIN. softmax_rows is the specialised
    # full-width path, so make it strictly cheaper at exactly 128 while this
    # kernel stays the choice below it.
    cost=lambda **params: (
        2.0 if softmax_query(params["shape"], params["dim"]).n == LANES else 1.0
    ),
)
