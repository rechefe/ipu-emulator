"""Row-softmax application harness (FP32 wide-vector mode).

Computes a numerically stable softmax over each of K rows of 128 FP32 elements
(any K >= 1), using the base-2 reformulation that matches the IPU's native
``exp2`` activation:

    softmax(x_i) = 2^(c*(x_i - xmax)) / SUM_j 2^(c*(x_j - xmax)),  c = log2(e)

so that ``2^(c * d) == e^d``. All scaling is done on the FP32 *vector* path
(``MULT.RC.VV`` / ``MULT.RC.VE``) because CR scalars are integer-only even in
wide mode (see docs/content/wide-vector-debug-mode.md). The constant ``c`` is
supplied as a resident 128-lane vector ``C_VEC``.

Pass structure (see the .asm for the cycle-level layout):

    Pass 1  (reduction):  maxvec[r]  = max_j (c * x[r,j])         -> staged XMEM
    Pass 2  (trip):       num[r,j]   = 2^(c*x[r,j] - maxvec[r])   -> NUM region
    Pass 3  (reduction):  sumvec[r]  = SUM_j num[r,j];  rvec = 1/sumvec  -> staged
    Pass 4  (trip):       out[r,j]   = num[r,j] * rvec[r]         -> OUT region

Only Passes 2 and 4 write a full result matrix; Passes 1 and 3 produce one
128-lane scalar vector each (maxvec / sumvec) that stays in R_ACC and is
staged once. See the project memory `softmax_rows_design` for the derivation
and the probe tests that validated each primitive.

Arbitrary row count: maxvec[r] / rvec[r] hold one scalar per row in a single
128-lane vector, so the kernel can carry the per-row bookkeeping for at most 128
rows at once. All four passes therefore run once per group of up to 128 rows.
The group size is computed EXACTLY in the kernel as min(128, rows_remaining) --
there is no padding, so a 7-row input runs exactly 7 rows and a 130-row input
runs 128 then 2. The three big regions (input / num / output) are sized to the
row count and placed back-to-back per instance (see _layout) so they don't
overlap for large inputs. Groups of 128 run at ~18 cyc/row.

Usage::

    from ipu_apps.softmax.softmax_rows import SoftmaxRowsApp

    app = SoftmaxRowsApp(
        inst_path="softmax_rows.bin",
        input_path="logits.bin",     # K * 512 bytes, FP32, row-major (any K)
        output_path="probs.bin",
        rows=500,
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

# -- Constants --------------------------------------------------------------

LANES = 128                  # elements per row (fixed by the 128-lane datapath)
ROW_BYTES = LANES * 4        # 512 bytes per FP32 row

# XMEM byte-address map. The three big regions (input / num / output) each hold
# padded_rows * 512 B and must not overlap; because K may exceed 128 rows their
# bases are computed per-instance (see _layout). INPUT_BASE is deliberately
# non-zero; literal 0 lives in its own CR (CR10). Module-level INPUT_BASE_ADDR /
# OUTPUT_BASE_ADDR are the K<=128 defaults kept for callers/tests that import
# them; the harness uses self.* bases.
INPUT_BASE_ADDR = 0x10000    # x[r]      (input logits, FP32) -- default base
NUM_BASE_ADDR = 0x20000      # num[r]    (numerators, Pass 2 output) -- default
OUTPUT_BASE_ADDR = 0x30000   # out[r]    (softmax, Pass 4 output) -- default
# Resident scalar vectors (1 chunk each) live above the big regions; placed
# per-instance in _layout so they clear the (variable-size) output region.
CVEC_ADDR = 0x40000          # C_VEC     (resident log2(e) constant, 1 row)
MAXVEC_ADDR = 0x40400        # maxvec    (staged per-row max, 1 row)
RVEC_ADDR = 0x40600          # rvec      (staged per-row 1/sum, 1 row)

NEG_ONE_BYTE = 0xFF          # CR low byte 0xFF -> signed -1 -> -1.0 in wide FP32

LOG2E = math.log2(math.e)    # c = 1.4426950408889634


class SoftmaxRowsApp(IpuApp):
    """Row-softmax over ROWS x 128 FP32 logits.

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Path to the input logits binary (rows * 512 B, FP32).
        output_path: Optional path to write the softmax output.
        rows:        Number of 128-element rows to process (any count >= 1;
                     default 128). Rows above 128 are processed in successive
                     groups whose size is computed exactly (the last group runs
                     only the rows that remain -- no padding).
    """

    def __init__(self, *, rows: int = 128, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.rows = int(rows)
        # maxvec/rvec hold one scalar per row in a single 128-lane vector, so the
        # kernel processes the rows in groups of at most 128. The group size is
        # exact (min(128, rows_left)) -- there is no padding.
        self.num_groups = (self.rows + LANES - 1) // LANES
        self._layout()

    def _layout(self) -> None:
        """Place the three big regions (each rows*512 B) back-to-back from
        INPUT_BASE_ADDR, then the resident scalar vectors above them. Region size
        scales with the row count, so the fixed 0x10000 spacing no longer
        suffices once rows > 128.
        """
        region = self.rows * ROW_BYTES
        # 64 KiB alignment keeps addresses readable and leaves slack.
        step = (region + 0xFFFF) & ~0xFFFF
        self.input_base = INPUT_BASE_ADDR
        self.num_base = self.input_base + step
        self.output_base = self.num_base + step
        scalars = self.output_base + step
        self.cvec_addr = scalars
        self.maxvec_addr = scalars + ROW_BYTES
        self.rvec_addr = scalars + 2 * ROW_BYTES

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires.

        ``wide_vector_quantize_output=False`` keeps lanes 4-byte FP32 through
        AAQ (AAQ is a no-op); ACTIVATE writes FP32 into POST_AAQ_REG and
        STR_POST_AAQ_REG drains the full 512 bytes.
        """
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        # dtype is otherwise unused on the FP32 wide path, but several helpers
        # branch on it; INT8 matches the existing wide-vector tests.
        state.dtype = DType.INT8
        return state

    def setup(self, state: "IpuState") -> None:
        # Input logits (only the real rows; padding rows stay zero-initialised).
        load_binary_to_xmem(
            state, self.input_path, self.input_base, ROW_BYTES, self.rows
        )

        # Resident constant vector (only c = log2(e) needs to be a vector;
        # it is fractional and cannot fit a CR scalar). The 1.0 identity-multiply
        # in Pass 3 uses CR1=1 via MULT.RC.VE, so no ONE_VEC is needed.
        cvec = struct.pack("<128f", *([LOG2E] * LANES))
        state.xmem.write_address(self.cvec_addr, cvec)

        # CR0 and CR1 are READ-ONLY: CR0 == 0 always, CR1 == 1 always.
        # We exploit both directly: CR0 is the zero source (cyclic index / init)
        # and CR1 == 1 is the 1.0 identity scalar for the Pass 3 multiply.
        # All writable CRs below hold integer addresses / strides / bounds, plus
        # the small int FP32 scalar CR11 = -1.
        state.regfile.set_cr(2, self.output_base)
        state.regfile.set_cr(3, self.cvec_addr)
        state.regfile.set_cr(4, self.num_base)
        state.regfile.set_cr(5, self.maxvec_addr)
        state.regfile.set_cr(6, self.rvec_addr)
        state.regfile.set_cr(7, ROW_BYTES)   # row stride (512)
        state.regfile.set_cr(8, 1)           # row-index increment
        state.regfile.set_cr(9, LANES)       # group cap = 128 rows (max per group)
        state.regfile.set_cr(10, self.input_base)  # input base (moved off read-only CR0)
        state.regfile.set_cr(11, NEG_ONE_BYTE)  # 0xFF -> -1.0 scalar (Pass 2 subtract)
        state.regfile.set_cr(12, LANES)      # 128: R1 base byte index for maxvec element select
        state.regfile.set_cr(13, self.rows)  # total row count (exact group sizing)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path, self.output_base, ROW_BYTES, self.rows
            )

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)
