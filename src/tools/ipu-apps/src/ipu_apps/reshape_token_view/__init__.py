"""Reshape-to-token-view test harness — MambaVisionMixer layout change.

Transposes an INT8 matrix between the two row-padded layouts
``MambaVisionMixer.forward`` alternates between:

- ``"t2c"``: token view ``(L, d_inner)`` -> channel view ``(d_inner, L)``,
  i.e. ``rearrange(xz, "b l d -> b d l")``
- ``"c2t"``: channel view ``(d_inner, L)`` -> token view ``(L, d_inner)``,
  i.e. ``rearrange(y, "b d l -> b l d")``

Both directions run the same program; only the CR registers differ.

The harness:

1. Writes the cyclic one-hot table the transpose reads from R_CYCLIC
2. Loads the source matrix into XMEM
3. Sets CR registers for base addresses, loop bounds and IpuState dtype
4. Runs the assembly program
5. Dumps the transposed matrix from XMEM

Usage::

    from ipu_apps.reshape_token_view import ReshapeTokenViewApp

    app = ReshapeTokenViewApp(
        inst_path="reshape.bin",
        inputs_path="t2c_in_int8.bin",
        output_path="out.bin",
        stage="stage3",
        direction="t2c",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import load_binary_to_xmem

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants --------------------------------------------------------------

#: Elements (and bytes, in INT8 mode) per XMEM row / cache line.
ROW_ELEMS = 128
ROW_BYTES = 128

#: ``(d_model, seq_len)`` for the two ``mamba_vision_T`` stages that contain
#: MambaVisionMixer blocks. Stage 3 is level 2 (dim 320, 14x14 window ->
#: L = 196); stage 4 is level 3 (dim 640, 7x7 window -> L = 49). The mixer is
#: built with ``expand=1``, so ``d_inner == d_model``.
STAGES = {
    "stage3": (320, 196),
    "stage4": (640, 49),
}

DIRECTIONS = ("t2c", "c2t")

#: XMEM .asm operands are ROW numbers (one row = 128 bytes), not byte
#: addresses, so the CR registers below carry rows; ``load_binary_to_xmem``
#: and ``xmem.write_address`` take byte addresses, hence ``* ROW_BYTES``.
#: Narrow (INT8) mode may only address the first 16384 rows.
ONEHOT_BASE_ROW = 0
ONEHOT_ROWS = 4
SRC_BASE_ROW = 1024
DST_BASE_ROW = 5120

#: STR_POST_AAQ_REG always writes all 512 B of the register. The kernel
#: stores to consecutive rows so each store overwrites the three zero rows
#: the previous one left, which keeps the output packed -- but the final
#: store still runs three rows past the end of the buffer.
STORE_SLACK_ROWS = 3


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def rows_for(length: int) -> int:
    """XMEM rows one zero-padded line of ``length`` elements occupies."""
    return ceil_div(length, ROW_ELEMS)


def pad_to_row(length: int) -> int:
    """``length`` rounded up to a whole number of XMEM rows."""
    return rows_for(length) * ROW_ELEMS


def parse_stage(stage: str) -> tuple[int, int]:
    """Parse a stage name into ``(d_model, seq_len)``."""
    try:
        return STAGES[stage]
    except KeyError:
        raise ValueError(
            f"Invalid stage '{stage}'. Supported: {', '.join(STAGES)}"
        ) from None


def _write_onehot_table(state: "IpuState") -> None:
    """Write the 512-element cyclic one-hot table into XMEM.

    Zero everywhere except element 128, so the 128-element window of
    R_CYCLIC starting at element ``128 - j`` is the one-hot vector e_j for
    every j in [0, 128). See ``reshape_token_view.asm`` for why that matters.
    """
    table = bytearray(ONEHOT_ROWS * ROW_BYTES)
    table[ROW_ELEMS] = 1
    state.xmem.write_address(ONEHOT_BASE_ROW * ROW_BYTES, table)


class ReshapeTokenViewApp(IpuApp):
    """Reshape-to-token-view application harness.

    Args:
        inst_path:   Path to assembled instruction binary.
        inputs_path: Path to the source matrix, packed as XMEM rows with its
            line count already padded up to a multiple of 128 (the inner
            loop is unconditionally 128 iterations long).
        output_path: Optional path to write output.
        stage:       ``'stage3'`` or ``'stage4'`` of ``mamba_vision_T``.
        direction:   ``'t2c'`` or ``'c2t'``.
        channels:    Channel count; defaults to ``d_inner``, which is what
            both mixer rearranges use. Pass ``d_model`` for a block-level
            tensor instead.
    """

    def __init__(
        self,
        *,
        stage: str = "stage3",
        direction: str = "t2c",
        channels: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if direction not in DIRECTIONS:
            raise ValueError(
                f"Invalid direction '{direction}'. Supported: {', '.join(DIRECTIONS)}"
            )
        self.inputs_path = Path(self.inputs_path)
        self.stage = stage
        self.direction = direction

        d_model, seq_len = parse_stage(stage)
        self.d_model = d_model
        self.seq_len = seq_len
        self.channels = d_model if channels is None else channels  # expand == 1

        # (M, N): the source has M lines of N elements, the result N of M.
        if direction == "t2c":
            self.src_lines, self.line_len = seq_len, self.channels
        else:
            self.src_lines, self.line_len = self.channels, seq_len

        self.src_rows_per_line = rows_for(self.line_len)     # SPL
        self.dst_rows_per_line = rows_for(self.src_lines)    # DPL
        self.src_rows = pad_to_row(self.src_lines) * self.src_rows_per_line
        self.dst_rows = self.line_len * self.dst_rows_per_line

    def setup(self, state: "IpuState") -> None:
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=ROW_ELEMS, partition=0)
        _write_onehot_table(state)

        # Zero first: source lines past ``src_lines`` must read as zero, and
        # the destination must not carry stale bytes into the golden compare.
        state.xmem.write_address(
            SRC_BASE_ROW * ROW_BYTES, bytearray(self.src_rows * ROW_BYTES)
        )
        state.xmem.write_address(
            DST_BASE_ROW * ROW_BYTES,
            bytearray((self.dst_rows + STORE_SLACK_ROWS) * ROW_BYTES),
        )
        load_binary_to_xmem(
            state, self.inputs_path, SRC_BASE_ROW * ROW_BYTES, ROW_BYTES, self.src_rows
        )

        # CR0=0 and CR1=1 permanently.
        state.regfile.set_cr(2, SRC_BASE_ROW)
        state.regfile.set_cr(3, DST_BASE_ROW)
        state.regfile.set_cr(4, self.dst_rows_per_line)
        state.regfile.set_cr(5, self.src_rows_per_line)
        state.regfile.set_cr(6, ROW_ELEMS + 1)  # R_CYCLIC window start, pre-incremented
        state.regfile.set_cr(7, self.line_len)  # output lines
        state.regfile.set_cr(8, self.dst_rows_per_line)  # output blocks per line
        state.regfile.set_cr(9, ROW_ELEMS)
        state.regfile.set_cr(10, ROW_ELEMS * self.src_rows_per_line)
        state.regfile.set_cr(11, ONEHOT_BASE_ROW)
        state.regfile.set_cr(13, 2 * ROW_ELEMS)  # R_CYCLIC load index 256
        state.regfile.set_cr(14, 3 * ROW_ELEMS)  # R_CYCLIC load index 384

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            data = state.xmem.read_address(
                DST_BASE_ROW * ROW_BYTES, self.dst_rows * ROW_BYTES
            )
            Path(self.output_path).write_bytes(bytes(data))
