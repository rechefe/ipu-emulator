"""Exercise harness for bootcamp drill 2 (conv): widen 3-tap conv to 5-tap.

Mirrors ConvApp's setup (same memory map, same INPUT_BASE/OUTPUT_BASE, same
CR2-CR6 conventions) but accepts a 5-tuple of tap weights and sets two more
tap-weight CRs (cr11 = w[-2], cr12 = w[2]) that the 5-tap .asm files need.

Usage::

    from ipu_apps.bootcamp.conv.exercise import Conv5App, SOLUTION_ASM_PATH, STUB_ASM_PATH

    app = Conv5App(
        inst_path="conv5.bin",
        input_path="rows.bin",
        output_path="output.bin",
        num_rows=4,
        weights=(1, 2, 3, 2, 1),  # w[-2], w[-1], w[0], w[1], w[2]
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_apps.base import IpuApp
from ipu_apps.bootcamp.conv import (
    LANES,
    INPUT_BASE,
    INPUT_BASE_ROW,
    OUTPUT_BASE,
    OUTPUT_BASE_ROW,
    OUTPUT_STRIDE_ROWS,
    ROW_BYTES_OUT,
    _to_signed_byte,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

_EXERCISE_DIR = Path(__file__).resolve().parent

STUB_ASM_PATH = _EXERCISE_DIR / "conv5_stub.asm"
SOLUTION_ASM_PATH = _EXERCISE_DIR / "conv5_solution.asm"

# 5-tap valid range: two edge lanes on each side read a wrapped/out-of-range
# neighbour (vs. one edge lane for the 3-tap drill) and are not real output.
VALID_LO = 2
VALID_HI = LANES - 2  # exclusive; valid range is [VALID_LO, VALID_HI)


class Conv5App(IpuApp):
    """5-tap variant of the conv drill: same memory map as ConvApp, plus
    two extra tap-weight CRs (cr11 = w[-2], cr12 = w[2]).

    Args:
        inst_path:   Path to assembled binary.
        input_path:  Path to input binary (num_rows * 128 bytes, INT8).
        output_path: Optional path to write INT32 output.
        num_rows:    Number of 128-element input rows (>= 1).
        weights:     5-tuple of signed INT8 tap weights
                     (w[-2], w[-1], w[0], w[1], w[2]).
    """

    ASM_PATH = SOLUTION_ASM_PATH

    def __init__(
        self,
        *,
        num_rows: int,
        input_path: str | Path,
        weights: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(input_path)

        if num_rows < 1:
            raise ValueError(f"num_rows must be >= 1, got {num_rows}")
        if len(weights) != 5:
            raise ValueError(f"weights must have exactly 5 taps, got {len(weights)}")

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

        w_l2, w_l1, w_c, w_r1, w_r2 = self.weights
        state.regfile.set_cr(8, _to_signed_byte(w_l1))
        state.regfile.set_cr(9, _to_signed_byte(w_c))
        state.regfile.set_cr(10, _to_signed_byte(w_r1))
        state.regfile.set_cr(11, _to_signed_byte(w_l2))
        state.regfile.set_cr(12, _to_signed_byte(w_r2))

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_bytes = self.num_rows * ROW_BYTES_OUT
            result = state.xmem.read_address(OUTPUT_BASE, total_bytes)
            Path(self.output_path).write_bytes(bytes(result))
