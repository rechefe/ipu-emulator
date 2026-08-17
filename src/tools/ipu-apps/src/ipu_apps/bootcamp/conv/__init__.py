"""Bootcamp drill 2/3: 1D single-channel 3-tap convolution (MULT + ACC).

Computes a "valid" (no padding, stride 1) 1D convolution with a 3-tap kernel
over ``num_rows`` independent rows of 128 INT8 elements each:

    out[r][p] = w[-1]*in[r][p-1] + w[0]*in[r][p] + w[1]*in[r][p+1]

for output positions p = 1 .. 126 (positions 0 and 127 would need an
out-of-bounds neighbour and are not produced -- see conv.asm's "WHY NO
MASKING" note). See conv.asm for the full walkthrough of how the
whole-vector shift-and-accumulate trick (MULT.RC.VE with rc_idx = -1/0/+1)
replaces an explicit per-position loop.

Layout: ``num_rows`` consecutive 128-byte INT8 input rows; INT32 output rows
(one STR_ACC_REG write = 512 bytes = 4 rows, same convention as
bootcamp/residual_add).

Usage::

    from ipu_apps.bootcamp.conv import ConvApp

    app = ConvApp(
        inst_path="conv.bin",
        input_path="rows.bin",
        output_path="output.bin",
        num_rows=4,
        weights=(1, 2, 1),
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

LANES = 128                  # elements per row (fixed by the datapath)
ROW_BYTES_IN = LANES         # INT8 input row: 128 bytes
ROW_BYTES_OUT = LANES * 4    # INT32 output row (r_acc is always 512 B/row)
# STR_ACC_REG always writes all 512 B of r_acc regardless of mode -- see
# bootcamp/residual_add's identical note.
OUTPUT_STRIDE_ROWS = ROW_BYTES_OUT // LANES

# Valid output positions: edges (lane 0 and lane LANES-1) read a wrapped or
# out-of-range neighbour and are not real conv output (see conv.asm).
VALID_LO = 1
VALID_HI = LANES - 1  # exclusive; valid range is [VALID_LO, VALID_HI)

# -- Byte-address memory map (harness-side only; converted to rows below) ----

INPUT_BASE = 0x0000
OUTPUT_BASE = 0x1000

INPUT_BASE_ROW = INPUT_BASE // LANES
OUTPUT_BASE_ROW = OUTPUT_BASE // LANES


def _to_signed_byte(value: int) -> int:
    """Reinterpret an int as an unsigned byte the CR low-byte scalar path expects."""
    return value & 0xFF


class ConvApp(IpuApp):
    """Bootcamp 1D 3-tap convolution: INT8 in, INT32 out, one row (128 lanes) at a time.

    Args:
        inst_path:   Path to assembled binary.
        input_path:  Path to input binary (num_rows * 128 bytes, INT8).
        output_path: Optional path to write INT32 output.
        num_rows:    Number of 128-element input rows (>= 1).
        weights:     3-tuple of signed INT8 tap weights (w[-1], w[0], w[1]).
    """

    ASM_PATH = Path(__file__).resolve().parent / "conv.asm"

    def __init__(
        self,
        *,
        num_rows: int,
        input_path: str | Path,
        weights: tuple[int, int, int] = (1, 1, 1),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(input_path)

        if num_rows < 1:
            raise ValueError(f"num_rows must be >= 1, got {num_rows}")
        if len(weights) != 3:
            raise ValueError(f"weights must have exactly 3 taps, got {len(weights)}")

        self.num_rows = num_rows
        self.weights = tuple(int(w) for w in weights)
        self.output_base = OUTPUT_BASE

    def setup(self, state: "IpuState") -> None:
        data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE, data)

        state.regfile.set_cr(2, INPUT_BASE_ROW)
        state.regfile.set_cr(3, OUTPUT_BASE_ROW)
        state.regfile.set_cr(4, 1)                    # input row step
        state.regfile.set_cr(5, OUTPUT_STRIDE_ROWS)    # output row step
        state.regfile.set_cr(6, self.num_rows)         # loop bound

        w_left, w_center, w_right = self.weights
        state.regfile.set_cr(8, _to_signed_byte(w_left))
        state.regfile.set_cr(9, _to_signed_byte(w_center))
        state.regfile.set_cr(10, _to_signed_byte(w_right))

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_bytes = self.num_rows * ROW_BYTES_OUT
            result = state.xmem.read_address(OUTPUT_BASE, total_bytes)
            Path(self.output_path).write_bytes(bytes(result))
