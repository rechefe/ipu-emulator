"""Bootcamp drill 1/3: residual add (INT8, LR + XMEM only).

Adds two INT8 tensors element-wise, ``num_vectors`` vectors of 128 elements
(1 row) each: ``out[v] = A[v] + B[v]``. See ``residual_add.asm`` for the full
walkthrough of the multiply-by-one trick and the pipelining rationale -- this
is the "no ISA surface beyond LR/LOAD/XMEM" drill in the bootcamp progression.

Layout (per tensor): ``num_vectors`` consecutive 128-byte rows, INT8 lanes.
Output rows are INT32 (r_acc is always 128 x INT32 = 512 bytes per row,
regardless of the INT8 input width).

Usage::

    from ipu_apps.bootcamp.residual_add import ResidualAddApp

    app = ResidualAddApp(
        inst_path="residual_add.bin",
        input_a_path="tensor_a.bin",
        input_b_path="tensor_b.bin",
        output_path="output.bin",
        num_vectors=4,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants ----------------------------------------------------------------

LANES = 128                 # elements per vector/row (fixed by the datapath)
ROW_BYTES_IN = LANES        # INT8 input row: 128 bytes
ROW_BYTES_OUT = LANES * 4   # INT32 output row (r_acc is always 512 B/row)
# STR_ACC_REG always writes all 512 B of r_acc regardless of mode, which
# spans 4 rows of 128 bytes each in narrow mode -- the stride between
# successive output vectors must match that, not the 1-row/128-byte input
# stride, or stores would overlap (see the .asm's stride note).
OUTPUT_STRIDE_ROWS = ROW_BYTES_OUT // LANES

# -- Byte-address memory map (harness-side only; converted to rows below) ----

INPUT_A_BASE = 0x0000
INPUT_B_BASE = 0x1000
OUTPUT_BASE = 0x2000

# .asm XMEM operands are ROW numbers, not byte addresses -- one row is 128
# bytes in narrow (INT8) mode. These *_BASE constants stay byte addresses
# because they only drive the harness's own xmem.write_address/read_address
# calls (byte-granular); the CR registers below feed the .asm's XMEM
# instructions and carry the row-number versions instead.
INPUT_A_BASE_ROW = INPUT_A_BASE // LANES
INPUT_B_BASE_ROW = INPUT_B_BASE // LANES
OUTPUT_BASE_ROW = OUTPUT_BASE // LANES


class ResidualAddApp(IpuApp):
    """Bootcamp residual add: INT8 + INT8 -> INT32, one row (128 lanes) per vector.

    Args:
        inst_path:    Path to assembled binary.
        input_a_path: Path to tensor A binary (num_vectors * 128 bytes, INT8).
        input_b_path: Path to tensor B binary (num_vectors * 128 bytes, INT8).
        output_path:  Optional path to write INT32 output.
        num_vectors:  Number of 128-element vectors (>= 1).
    """

    ASM_PATH = Path(__file__).resolve().parent / "residual_add.asm"

    def __init__(
        self,
        *,
        num_vectors: int,
        input_a_path: str | Path,
        input_b_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_a_path = Path(input_a_path)
        self.input_b_path = Path(input_b_path)

        if num_vectors < 1:
            raise ValueError(f"num_vectors must be >= 1, got {num_vectors}")

        self.num_vectors = num_vectors
        self.output_base = OUTPUT_BASE

    def setup(self, state: "IpuState") -> None:
        input_a = self.input_a_path.read_bytes()
        state.xmem.write_address(INPUT_A_BASE, input_a)

        input_b = self.input_b_path.read_bytes()
        state.xmem.write_address(INPUT_B_BASE, input_b)

        # cr0/cr1 are hardware-reserved (always read 0 / 1); the asm reuses
        # cr1 directly as the "multiply by 1" identity scalar.
        state.regfile.set_cr(2, INPUT_A_BASE_ROW)
        state.regfile.set_cr(3, INPUT_B_BASE_ROW)
        state.regfile.set_cr(4, OUTPUT_BASE_ROW)
        state.regfile.set_cr(5, 1)                 # input row step: 1 row per vector
        state.regfile.set_cr(6, self.num_vectors)  # loop bound
        state.regfile.set_cr(7, OUTPUT_STRIDE_ROWS)  # output row step: 4 rows per vector

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_bytes = self.num_vectors * ROW_BYTES_OUT
            result = state.xmem.read_address(OUTPUT_BASE, total_bytes)
            Path(self.output_path).write_bytes(bytes(result))
