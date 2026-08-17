"""Bootcamp drill 3/3: row-softmax, single tile (rows <= 128), FP32 wide-vector.

Computes a numerically-stable softmax over each of ``rows`` (<= 128) rows of
128 FP32 elements, using the base-2 reformulation that matches the IPU's
native ``exp2`` activation:

    softmax(x)_j = 2^(c*(x_j - xmax)) / SUM_k 2^(c*(x_k - xmax)),  c = log2(e)

See ``softmax.asm`` for the full 4-pass walkthrough (max / exp / sum /
normalize) and why this drill stops at 128 rows (one "group" -- no
cross-group loop, unlike the project's production ``softmax_rows`` app).

Usage::

    from ipu_apps.bootcamp.softmax import SoftmaxApp

    app = SoftmaxApp(
        inst_path="softmax.bin",
        input_path="logits.bin",   # rows * 512 bytes, FP32, row-major
        output_path="probs.bin",
        rows=16,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import load_binary_to_xmem, dump_xmem_to_binary
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    pass

# -- Constants ----------------------------------------------------------------

LANES = 128               # elements per row (fixed by the 128-lane datapath)
ROW_BYTES = LANES * 4     # 512 bytes per FP32 row (wide-vector debug mode)

# -- Byte-address memory map (harness-side only; converted to rows below) ----
#
# .asm XMEM operands are ROW numbers (one row = 128 elements = ROW_BYTES in
# wide-vector-debug mode), not byte addresses -- same convention as
# bootcamp/residual_add and bootcamp/conv. The *_BASE constants stay byte
# addresses because they only drive the harness's own xmem.write_address/
# read_address calls; the CR registers below feed the .asm's XMEM
# instructions and carry the row-number versions instead.

INPUT_BASE = 0x10000     # x[r]    (input logits, FP32)
NUM_BASE = 0x20000        # num[r]  (Pass 2 numerators)
OUTPUT_BASE = 0x30000     # out[r]  (Pass 4 softmax output)
CVEC_ADDR = 0x40000       # C_VEC   (resident log2(e) constant, 1 row)
MAXVEC_ADDR = 0x40200     # maxvec  (staged per-row max, 1 row)
RVEC_ADDR = 0x40400       # rvec    (staged per-row 1/sum, 1 row)

LOG2E = math.log2(math.e)  # c = 1.4426950408889634


class SoftmaxApp(IpuApp):
    """Bootcamp softmax: single-tile row-softmax over up to 128 rows of 128 FP32 logits.

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Path to the input logits binary (rows * 512 B, FP32).
        output_path: Optional path to write the softmax output.
        rows:        Number of 128-element rows to process (1 <= rows <= 128
                     -- this drill is deliberately single-tile; see the .asm
                     header for why).
    """

    ASM_PATH = Path(__file__).resolve().parent / "softmax.asm"

    def __init__(self, *, rows: int = 128, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.rows = int(rows)
        if not (1 <= self.rows <= LANES):
            raise ValueError(
                f"rows must be in [1, {LANES}] -- this drill is single-tile "
                f"(no cross-group loop); got {self.rows}"
            )
        self.output_base = OUTPUT_BASE

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires.

        ``wide_vector_quantize_output=False`` keeps lanes 4-byte FP32 through
        AAQ (AAQ itself is a no-op in this mode); ACTIVATE writes FP32 into
        POST_AAQ_REG and STR_POST_AAQ_REG drains the full 512 bytes.
        """
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        # dtype is otherwise unused on the FP32 wide path, but several
        # helpers branch on it; INT8 matches the existing wide-vector apps.
        state.dtype = DType.INT8
        return state

    def setup(self, state: "IpuState") -> None:
        load_binary_to_xmem(state, self.input_path, INPUT_BASE, ROW_BYTES, self.rows)

        # Resident constant vector: c = log2(e) is fractional and can't fit a
        # CR scalar, so it lives as a 128-lane FP32 vector in XMEM instead.
        cvec = struct.pack("<128f", *([LOG2E] * LANES))
        state.xmem.write_address(CVEC_ADDR, cvec)

        # CR0/CR1 are READ-ONLY hardware constants (always 0 / 1); the asm
        # reuses both directly (cr0 as the zero/init source, cr1 as the 1.0
        # identity scalar for Pass 2's subtract and Pass 3's copy-multiply).
        state.regfile.set_cr(2, OUTPUT_BASE // ROW_BYTES)
        state.regfile.set_cr(3, CVEC_ADDR // ROW_BYTES)
        state.regfile.set_cr(4, NUM_BASE // ROW_BYTES)
        state.regfile.set_cr(5, MAXVEC_ADDR // ROW_BYTES)
        state.regfile.set_cr(6, RVEC_ADDR // ROW_BYTES)
        state.regfile.set_cr(7, 1)          # row step: 1 row per element/row
        state.regfile.set_cr(9, LANES)      # R1 byte-index base for maxvec element select
        state.regfile.set_cr(10, INPUT_BASE // ROW_BYTES)
        state.regfile.set_cr(13, self.rows)  # exact row count (<= 128, no padding)

        # AGG.*/ACTIVATE.QUANTIZE read the active-lane count from the named
        # dstructure CR's valid_elements field -- the asm names CR15 on every
        # AGG/ACTIVATE.QUANTIZE, so set CR15.valid_elements = 128 to process
        # the full FP32 row (all 128 logits in every row participate).
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(state, self.output_path, OUTPUT_BASE, ROW_BYTES, self.rows)

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)
