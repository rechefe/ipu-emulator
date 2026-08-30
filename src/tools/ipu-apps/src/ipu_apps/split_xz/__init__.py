"""Split-to-x-and-z test harness — MambaVisionMixer branch split.

Implements ``x, z = xz.chunk(2, dim=1)`` from ``MambaVisionMixer.forward``
on the channel-view tensor ``(d_inner, L)``: the first ``d_inner // 2``
channel lines become x, the rest become z.

The harness:

1. Loads the channel-view xz buffer into XMEM
2. Sets CR registers for base addresses, the row count and IpuState dtype
3. Runs the assembly program
4. Dumps the x and z buffers from XMEM

Usage::

    from ipu_apps.split_xz import SplitXzApp

    app = SplitXzApp(
        inst_path="split_xz.bin",
        inputs_path="xz_in_int8.bin",
        output_path="x.bin",
        output_z_path="z.bin",
        stage="stage3",
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

ROW_ELEMS = 128
ROW_BYTES = 128

#: ``(d_model, seq_len)`` for the two ``mamba_vision_T`` stages that contain
#: MambaVisionMixer blocks. The mixer is built with ``expand=1``, so
#: ``d_inner == d_model`` and each branch is ``d_model // 2`` channels.
STAGES = {
    "stage3": (320, 196),
    "stage4": (640, 49),
}

#: XMEM row numbers (one row = 128 bytes). Narrow (INT8) mode may only
#: address the first 16384 rows.
SRC_BASE_ROW = 1024
DSTX_BASE_ROW = 5120
DSTZ_BASE_ROW = 7168

#: STR_POST_AAQ_REG writes 512 B; consecutive stores keep the buffer packed
#: but the last one runs three rows past the end.
STORE_SLACK_ROWS = 3


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def parse_stage(stage: str) -> tuple[int, int]:
    """Parse a stage name into ``(d_model, seq_len)``."""
    try:
        return STAGES[stage]
    except KeyError:
        raise ValueError(
            f"Invalid stage '{stage}'. Supported: {', '.join(STAGES)}"
        ) from None


class SplitXzApp(IpuApp):
    """Split-to-x-and-z application harness.

    Args:
        inst_path:     Path to assembled instruction binary.
        inputs_path:   Path to the channel-view xz buffer.
        output_path:   Optional path to write the x branch.
        output_z_path: Optional path to write the z branch.
        stage:         ``'stage3'`` or ``'stage4'`` of ``mamba_vision_T``.
    """

    def __init__(self, *, stage: str = "stage3", output_z_path=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inputs_path = Path(self.inputs_path)
        self.output_z_path = Path(output_z_path) if output_z_path else None
        self.stage = stage

        d_model, seq_len = parse_stage(stage)
        self.d_model = d_model
        self.seq_len = seq_len
        self.d_inner = d_model  # expand == 1
        self.half = self.d_inner // 2

        self.rows_per_channel = ceil_div(seq_len, ROW_ELEMS)
        self.half_rows = self.half * self.rows_per_channel
        self.src_rows = self.d_inner * self.rows_per_channel

    def setup(self, state: "IpuState") -> None:
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=ROW_ELEMS, partition=0)

        load_binary_to_xmem(
            state, self.inputs_path, SRC_BASE_ROW * ROW_BYTES, ROW_BYTES, self.src_rows
        )
        # The loop prefetches one row past the last it copies; keep that read
        # on defined memory, and clear both destinations.
        state.xmem.write_address((SRC_BASE_ROW + self.src_rows) * ROW_BYTES,
                                 bytearray(ROW_BYTES))
        for base in (DSTX_BASE_ROW, DSTZ_BASE_ROW):
            state.xmem.write_address(
                base * ROW_BYTES,
                bytearray((self.half_rows + STORE_SLACK_ROWS) * ROW_BYTES),
            )

        # CR0=0 and CR1=1 permanently; CR1 doubles as the multiply-by-one
        # scalar the copy loops use, so no constants row is needed.
        state.regfile.set_cr(2, SRC_BASE_ROW)
        state.regfile.set_cr(3, DSTX_BASE_ROW)
        state.regfile.set_cr(4, DSTZ_BASE_ROW)
        state.regfile.set_cr(5, self.half_rows)

    def teardown(self, state: "IpuState") -> None:
        size = self.half_rows * ROW_BYTES
        if self.output_path is not None:
            data = state.xmem.read_address(DSTX_BASE_ROW * ROW_BYTES, size)
            Path(self.output_path).write_bytes(bytes(data))
        if self.output_z_path is not None:
            data = state.xmem.read_address(DSTZ_BASE_ROW * ROW_BYTES, size)
            Path(self.output_z_path).write_bytes(bytes(data))
