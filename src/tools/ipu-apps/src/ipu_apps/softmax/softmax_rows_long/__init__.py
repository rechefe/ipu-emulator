"""Long-row softmax harness (FP32 wide-vector mode).

Mix of ``softmax_rows`` (full 128-wide rows) and ``softmax_rows_partial``
(sub-128 rows): handles rows whose element count ``N > 128`` and is **not**
divisible by 128. Each row is ``full_chunks = N // 128`` full 128-element
chunks plus one tail of ``tail = N % 128`` elements (``tail > 0``).

    softmax(x_i) = 2^(c*(x_i - xmax)) / SUM_j 2^(c*(x_j - xmax)),  c = log2(e)

so the IPU's native ``exp2`` activation applies directly (``2^(c*d) == e^d``).
``c = log2(e)`` rides in a resident 128-element vector ``C_VEC``; the 1.0 scalar is
the read-only CR1.

Per-row scalars (``maxvec[r]`` / ``rvec[r]``) hold one value per row in a single
128-element vector, so this version processes up to 128 rows in one group (no group
loop yet -- see ``softmax_rows`` for the >128-row group machinery).

The novelty vs the two base apps is the **cross-chunk reduction**: a single
row's max (Pass 1) and sum (Pass 3) now span ``cpr = full_chunks + 1`` chunks,
so each pass keeps a *running* per-row scalar -- ``AGG.*.FIRST`` on chunk 0 then
``AGG.*`` (running) on chunks 1.. -- combining all of a row's chunks into one
slot before the row's normalisation.

Layout (row-contiguous, chunk-padded): row ``r`` occupies ``cpr`` consecutive
512 B chunks at ``BASE + r*cpr*512``; the tail chunk's unused elements
(``tail..127``) are zero-padded. The full chunks reduce with
``valid_elements=128`` (CR15); the tail chunk reduces with
``valid_elements=tail`` (CR8) -- the dual-dstructure-CR trick from
``softmax_rows_partial``.

Usage::

    from ipu_apps.softmax.softmax_rows_long import SoftmaxRowsLongApp

    app = SoftmaxRowsLongApp(
        inst_path="softmax_rows_long.bin",
        input_path="logits.bin",     # rows * N * 4 bytes, FP32, row-major
        output_path="probs.bin",
        rows=8,
        n=300,                        # N>128, N%128 != 0
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

LANES = 128                  # elements per chunk (fixed by the 128-element datapath)
CHUNK_BYTES = LANES * 4      # 512 bytes per FP32 chunk

INPUT_BASE_ADDR = 0x10000    # x   (input logits, FP32)
NUM_BASE_ADDR = 0x80000      # num (numerators, Pass 2 output)
OUTPUT_BASE_ADDR = 0xF0000   # out (softmax, Pass 4 output)
# Resident scalar vectors (1 chunk each) above the big regions.
CVEC_ADDR = 0x160000         # C_VEC  (resident log2(e) constant, 1 chunk)
MAXVEC_ADDR = 0x160400       # maxvec (staged per-row max, 1 chunk)
RVEC_ADDR = 0x160600         # rvec   (staged per-row 1/sum, 1 chunk)

LOG2E = math.log2(math.e)    # c = 1.4426950408889634


class SoftmaxRowsLongApp(IpuApp):
    """Row-softmax over ROWS rows of N FP32 logits, N>128 and N%128 != 0.

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Path to the input logits binary (rows * N * 4 B, FP32,
                     row-major).
        output_path: Optional path to write the softmax output (same shape).
        rows:        Number of rows (1..128 -- single group).
        n:           Elements per row. Must satisfy n > 128 and n % 128 != 0.
    """

    def __init__(self, *, rows: int, n: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.rows = int(rows)
        self.n = int(n)

        # Delegate to the registry declaration rather than restating the bounds:
        # SPEC.supports is the single source of truth for this kernel's domain,
        # so the guard and the router can never disagree.
        SPEC.guard(shape=(self.rows, self.n), dim=1)

        self.full_chunks = self.n // 128            # number of full 128-wide chunks
        self.tail = self.n % 128                    # tail element count (0..127)
        # A tail chunk exists only when n isn't a clean multiple of 128. When
        # tail == 0 the row is exactly full_chunks whole chunks: emitting the
        # usual +1 chunk would read a phantom chunk past the row. The .asm
        # needs no change for this case -- Passes 2/4 simply loop cpr chunks,
        # and Passes 1/3 run their tail block with valid_elements=0 (CR8),
        # which makes the running AGG.MAX / AGG.SUM exact no-ops.
        self.chunks_per_row = self.full_chunks + (1 if self.tail else 0)

        self._layout()

    def _layout(self) -> None:
        """Size the three big regions to rows*cpr*512 B and place them
        back-to-back from INPUT_BASE_ADDR; resident scalar vectors above.
        """
        region = self.rows * self.chunks_per_row * CHUNK_BYTES
        step = (region + 0xFFFF) & ~0xFFFF       # 64 KiB align
        self.input_base = INPUT_BASE_ADDR
        self.num_base = self.input_base + step
        self.output_base = self.num_base + step
        scalars = self.output_base + step
        self.cvec_addr = scalars
        self.maxvec_addr = scalars + CHUNK_BYTES
        self.rvec_addr = scalars + 2 * CHUNK_BYTES

    # -- packing ------------------------------------------------------------

    def _pack_input(self) -> bytes:
        """Read row-major (rows x N) float32 from input_path and pack into the
        chunk-padded layout: row r, chunk c at offset (r*cpr + c)*512; the tail
        chunk's elements [tail..128) stay zero.
        """
        raw = self.input_path.read_bytes()
        flat = struct.unpack(f"<{self.rows * self.n}f", raw[: self.rows * self.n * 4])
        packed = bytearray(self.rows * self.chunks_per_row * CHUNK_BYTES)
        for r in range(self.rows):
            row = flat[r * self.n:(r + 1) * self.n]
            for c in range(self.chunks_per_row):
                lo = c * 128
                hi = min(lo + 128, self.n)          # tail chunk: hi = N
                base = (r * self.chunks_per_row + c) * CHUNK_BYTES
                struct.pack_into(f"<{hi - lo}f", packed, base, *row[lo:hi])
        return bytes(packed)

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        state.dtype = DType.INT8  # mode byte; wide flag overrides element width
        return state

    def setup(self, state: "IpuState") -> None:
        state.xmem.write_address(self.input_base, self._pack_input())

        cvec = struct.pack("<128f", *([LOG2E] * LANES))
        state.xmem.write_address(self.cvec_addr, cvec)

        # CR0 == 0, CR1 == 1 are READ-ONLY. CR1 (=1.0) is the identity scalar
        # (Pass 3/4) and, with ACC.SUB, the Pass 2 subtract; CR1 also serves as
        # the +1 loop increment.
        # .asm XMEM operands are ROW numbers (one row = CHUNK_BYTES), not byte
        # addresses -- see issue #179 -- so all base/stride CRs below are rows.
        state.regfile.set_cr(2, self.output_base // CHUNK_BYTES)
        state.regfile.set_cr(3, self.cvec_addr // CHUNK_BYTES)
        state.regfile.set_cr(4, self.num_base // CHUNK_BYTES)
        state.regfile.set_cr(5, self.maxvec_addr // CHUNK_BYTES)
        state.regfile.set_cr(6, self.rvec_addr // CHUNK_BYTES)
        state.regfile.set_cr(7, 1)                      # chunk stride: advance one row per chunk
        # CR8 = tail. Decoded as a dstructure its valid_elements = tail, so the
        # tail chunk's AGG.MAX/AGG.SUM reduce only the first `tail` elements (the
        # dual-dstructure-CR trick: CR15=128 for full chunks, CR8=tail here).
        state.regfile.set_cr(8, self.tail)
        state.regfile.set_cr(9, LANES)                  # 128 (maxvec R1 byte base)
        state.regfile.set_cr(10, self.input_base // CHUNK_BYTES)  # input base row
        state.regfile.set_cr(11, self.rows)             # row count (loop bound)
        state.regfile.set_cr(12, self.full_chunks)      # full-chunk loop bound
        state.regfile.set_cr(13, self.chunks_per_row)   # cpr (row stride in chunks)
        state.regfile.set_cr(14, self.chunks_per_row)   # row stride (rows): advance cpr rows per logical row

        # CR15 = full-chunk dstructure: valid_elements = 128.
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(
            self.output_base, self.rows * self.chunks_per_row * CHUNK_BYTES
        )
        # Un-pack: drop each tail chunk's padding elements, emit row-major rows x N.
        out = bytearray(self.rows * self.n * 4)
        for r in range(self.rows):
            for c in range(self.chunks_per_row):
                lo = c * 128
                hi = min(lo + 128, self.n)
                src = (r * self.chunks_per_row + c) * CHUNK_BYTES
                dst = (r * self.n + lo) * 4
                out[dst:dst + (hi - lo) * 4] = raw[src:src + (hi - lo) * 4]
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
    if q.n <= LANES:
        return no(
            f"splits a row across several {LANES}-element chunks, so it needs a "
            f"row longer than {LANES}; this row has {q.n}"
        )
    return yes()


def _build(**params):
    q = softmax_query(params["shape"], params["dim"])
    return {"n": q.n, "rows": q.rows}


def _explain(**params):
    q = softmax_query(params["shape"], params["dim"])
    full, tail = divmod(q.n, LANES)
    shape = (
        f"{full} full chunks + {tail}-element tail"
        if tail else
        f"exactly {full} full chunks (no tail chunk)"
    )
    return f"n ({q.n}) > {LANES}: {shape}, reduced with a running cross-chunk AGG."


SPEC = KernelSpec(
    name="softmax_rows_long",
    op="softmax",
    variant="rows_long",
    app_class=SoftmaxRowsLongApp,
    asm="softmax_rows_long.asm",
    # Every callback below indexes these, so the registry checks them first:
    # an omitted parameter is then a refusal that names what is missing.
    requires=("shape", "dim"),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: softmax_query(params["shape"], params["dim"]).bundle,
    cost=lambda **params: 1.0,
)
