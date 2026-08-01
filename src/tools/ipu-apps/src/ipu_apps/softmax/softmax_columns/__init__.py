"""Column-softmax harness (FP32 wide-vector mode).

Softmax taken **down each column**: element ``x[r, col]`` is normalised against
all rows of the same column, ``{x[*, col]}``. This is the transpose of the
``softmax_rows`` family -- there the reduction collapses the 128 lanes of one row
to a scalar; here every lane is an *independent* column and the reduction runs
*across the rows* (across vectors), staying in-lane the whole time. So there is
**no AGG and no fan-out**: the per-column max/sum are full 128-lane vectors, and
ACC.MAX / ACC.ADD do the running reduce down the rows.

    softmax(x[r,col]) = 2^(c*(x[r,col]-cmax[col])) / SUM_r 2^(c*(x[r,col]-cmax[col]))

with ``c = log2(e)`` resident in a 128-lane vector ``C_VEC`` (``2^(c*d)==e^d``,
matching the IPU's native ``exp2`` activation).

Layout (v1 -- width >= 128, any value):
  Each row has width ``W`` padded up to the next multiple of 128, spanning
  ``cpr = ceil(W/128)`` consecutive 512 B chunks (no upper width bound). Row
  ``r``, chunk ``c`` lives at ``(r*cpr + c)*512``. A width that is not a multiple
  of 128 is padded with **0.0**; those filler lanes are *separate columns* (never
  part of a real column's reduce), computed harmlessly and dropped on read-back.
  ``rows`` is an ordinary loop bound -- any count works (no group cap, unlike the
  row apps, because the per-column scalars are full vectors, not packed-per-row).

Per-chunk-column resident vectors ``cmax[c]`` / ``rvec[c]`` (one full 128-lane
vector each) hold the column max / 1-over-sum for chunk-column ``c``. The four
passes loop **outer over chunk c, inner over rows r** so each pass's running ACC
reduce sweeps all rows for one chunk-column before moving on.

    Pass 1 (reduce):  cmax[c]  = max_r (c * x[r,c])           -> staged XMEM
    Pass 2 (trip):    num[r,c] = 2^(c*x[r,c] - cmax[c])       -> NUM region
    Pass 3 (reduce):  sum[c]   = SUM_r num[r,c]; rvec=1/sum    -> staged XMEM
    Pass 4 (trip):    out[r,c] = num[r,c] * rvec[c]           -> OUT region

Usage::

    from ipu_apps.softmax.softmax_columns import SoftmaxColumnsApp

    app = SoftmaxColumnsApp(
        inst_path="softmax_columns.bin",
        input_path="logits.bin",      # rows * width * 4 B, FP32, row-major
        output_path="probs.bin",
        rows=64,
        width=128,                    # pow2, 128..256 (padded up if needed)
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

if TYPE_CHECKING:
    pass

# -- Constants --------------------------------------------------------------

LANES = 128                  # elements per chunk (fixed by the 128-lane datapath)
CHUNK_BYTES = LANES * 4      # 512 bytes per FP32 chunk
MIN_WIDTH = 128              # v1: width >= 128 (one or more whole chunks per row)

INPUT_BASE_ADDR = 0x10000    # x   (input logits, FP32)
# num / output bases and the resident vectors are placed per-instance (_layout).

LOG2E = math.log2(math.e)    # c = 1.4426950408889634


class SoftmaxColumnsApp(IpuApp):
    """Column-softmax over ``rows`` x ``width`` FP32 logits (width >= 128).

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Path to the input logits binary (rows * width * 4 B, FP32,
                     row-major). ``width`` here is the *real* (unpadded) width.
        output_path: Optional path to write the softmax output (same shape).
        rows:        Number of rows (>= 1; any count -- ordinary loop bound).
        width:       Real elements per row (>= 128, no upper bound). Padded up to
                     the next multiple of 128 for the on-chip layout, spanning
                     ``cpr = ceil(width/128)`` chunks; padding lanes are dropped on
                     read-back. (e.g. 300 -> 384/3 chunks, 460 -> 512/4 chunks.)
    """

    def __init__(self, *, rows: int, width: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.rows = int(rows)
        self.width = int(width)

        if self.rows < 1:
            raise ValueError(f"rows ({self.rows}) must be >= 1")
        if self.width < MIN_WIDTH:
            raise ValueError(
                f"width ({self.width}) must be >= {MIN_WIDTH} in v1 "
                f"(sub-128 packed widths are a future extension)"
            )

        # Pad up to the next whole multiple of 128 so the row tiles into whole
        # 512 B chunks; cpr grows freely (no upper width bound).
        self.chunks_per_row = (self.width + LANES - 1) // LANES   # cpr = ceil(width/128)
        self.padded_width = self.chunks_per_row * LANES
        self._layout()

    def _layout(self) -> None:
        """Size input / num / output regions to rows*cpr*512 B, place them
        back-to-back; the resident per-chunk-column vectors (C_VEC, cmax, rvec)
        live above. cmax/rvec hold ``cpr`` chunks each (one per chunk-column).
        """
        region = self.rows * self.chunks_per_row * CHUNK_BYTES
        step = (region + 0xFFFF) & ~0xFFFF       # 64 KiB align
        self.input_base = INPUT_BASE_ADDR
        self.num_base = self.input_base + step
        self.output_base = self.num_base + step
        scalars = self.output_base + step
        self.cvec_addr = scalars                              # 1 chunk
        self.cmax_addr = scalars + CHUNK_BYTES                # cpr chunks
        self.rvec_addr = self.cmax_addr + self.chunks_per_row * CHUNK_BYTES

    # -- packing ------------------------------------------------------------

    def _pack_input(self) -> bytes:
        """Read row-major (rows x width) float32 and pack into the chunk layout:
        row r, chunk c at offset (r*cpr + c)*512. Lanes >= real width within the
        last chunk stay zero (the pow2 width padding).
        """
        raw = self.input_path.read_bytes()
        flat = struct.unpack(f"<{self.rows * self.width}f", raw[: self.rows * self.width * 4])
        packed = bytearray(self.rows * self.chunks_per_row * CHUNK_BYTES)
        for r in range(self.rows):
            row = flat[r * self.width:(r + 1) * self.width]
            for c in range(self.chunks_per_row):
                lo = c * LANES
                hi = min(lo + LANES, self.width)
                if hi <= lo:
                    continue
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
        # (Pass 2 subtract via ACC.SUB, Pass 3 sum-multiply) and the +1 loop
        # increment. All lanes are live for every op, so every AGG/ACTIVATE.
        # QUANTIZE names CR15 with valid_elements = 128 (no dual-CR trick here:
        # column softmax never partial-reduces a vector).
        # .asm XMEM operands are ROW numbers (one row = CHUNK_BYTES), not byte
        # addresses -- see issue #179 -- so all base/stride CRs below are rows.
        state.regfile.set_cr(2, self.output_base // CHUNK_BYTES)
        state.regfile.set_cr(3, self.cvec_addr // CHUNK_BYTES)
        state.regfile.set_cr(4, self.num_base // CHUNK_BYTES)
        state.regfile.set_cr(5, self.cmax_addr // CHUNK_BYTES)
        state.regfile.set_cr(6, self.rvec_addr // CHUNK_BYTES)
        state.regfile.set_cr(7, 1)                            # chunk stride: advance one row per chunk
        state.regfile.set_cr(9, self.chunks_per_row)          # row stride (rows)
        state.regfile.set_cr(10, self.input_base // CHUNK_BYTES)  # input base row
        state.regfile.set_cr(11, self.rows)                   # row loop bound
        state.regfile.set_cr(13, self.chunks_per_row)         # chunk-column loop bound

        # CR15 = dstructure: valid_elements = 128 (full chunk, every lane live).
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(
            self.output_base, self.rows * self.chunks_per_row * CHUNK_BYTES
        )
        # Un-pack: drop each row's pow2 width-padding lanes, emit row-major
        # rows x width.
        out = bytearray(self.rows * self.width * 4)
        for r in range(self.rows):
            for c in range(self.chunks_per_row):
                lo = c * LANES
                hi = min(lo + LANES, self.width)
                if hi <= lo:
                    continue
                src = (r * self.chunks_per_row + c) * CHUNK_BYTES
                dst = (r * self.width + lo) * 4
                out[dst:dst + (hi - lo) * 4] = raw[src:src + (hi - lo) * 4]
        Path(self.output_path).write_bytes(bytes(out))

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)
