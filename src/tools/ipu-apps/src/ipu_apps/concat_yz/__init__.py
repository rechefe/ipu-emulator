"""Concat-y-and-z test harness — MambaVisionMixer branch merge.

Implements ``y = torch.cat([y, z], dim=1)`` from ``MambaVisionMixer.forward``:
the selective-scan output y and the gated branch z, both channel-view
``(d_inner // 2, L)``, are stacked into one ``(d_inner, L)`` buffer that
``out_proj`` then consumes.

The harness:

1. Loads the y and z channel-view buffers into XMEM
2. Sets CR registers for base addresses, the row count and IpuState dtype
3. Runs the assembly program
4. Dumps the concatenated buffer from XMEM

Usage::

    from ipu_apps.concat_yz import ConcatYzApp

    app = ConcatYzApp(
        inst_path="concat_yz.bin",
        inputs_path="y_in_int8.bin",
        z_path="z_in_int8.bin",
        output_path="yz.bin",
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
#: MambaVisionMixer blocks. ``expand=1``, so ``d_inner == d_model``.
STAGES = {
    "stage3": (320, 196),
    "stage4": (640, 49),
}

#: XMEM row numbers (one row = 128 bytes). Narrow (INT8) mode may only
#: address the first 16384 rows.
SRCY_BASE_ROW = 1024
SRCZ_BASE_ROW = 3072
DST_BASE_ROW = 5120

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


class ConcatYzApp(IpuApp):
    """Concat-y-and-z application harness.

    Args:
        inst_path:   Path to assembled instruction binary.
        inputs_path: Path to the y buffer (channel view).
        z_path:      Path to the z buffer (channel view).
        output_path: Optional path to write the concatenated result.
        stage:       ``'stage3'`` or ``'stage4'`` of ``mamba_vision_T``.
    """

    def __init__(self, *, stage: str = "stage3", z_path=None, **kwargs) -> None:
        super().__init__(**kwargs)
        if z_path is None:
            raise ValueError("concat_yz needs both inputs: pass z_path=...")
        self.inputs_path = Path(self.inputs_path)
        self.z_path = Path(z_path)
        self.stage = stage

        d_model, seq_len = parse_stage(stage)
        self.d_model = d_model
        self.seq_len = seq_len
        self.d_inner = d_model  # expand == 1
        self.half = self.d_inner // 2

        self.rows_per_channel = ceil_div(seq_len, ROW_ELEMS)
        self.half_rows = self.half * self.rows_per_channel
        self.dst_rows = self.d_inner * self.rows_per_channel

    def setup(self, state: "IpuState") -> None:
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=ROW_ELEMS, partition=0)

        load_binary_to_xmem(
            state, self.inputs_path, SRCY_BASE_ROW * ROW_BYTES, ROW_BYTES, self.half_rows
        )
        load_binary_to_xmem(
            state, self.z_path, SRCZ_BASE_ROW * ROW_BYTES, ROW_BYTES, self.half_rows
        )
        # Both loops prefetch one row past their last; keep those defined.
        for base in (SRCY_BASE_ROW, SRCZ_BASE_ROW):
            state.xmem.write_address((base + self.half_rows) * ROW_BYTES,
                                     bytearray(ROW_BYTES))
        state.xmem.write_address(
            DST_BASE_ROW * ROW_BYTES,
            bytearray((self.dst_rows + STORE_SLACK_ROWS) * ROW_BYTES),
        )

        # CR0=0 and CR1=1 permanently; CR1 doubles as the multiply-by-one
        # scalar the copy loops use, so no constants row is needed.
        state.regfile.set_cr(2, SRCY_BASE_ROW)
        state.regfile.set_cr(3, DST_BASE_ROW)
        state.regfile.set_cr(4, SRCZ_BASE_ROW)
        state.regfile.set_cr(5, self.half_rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            data = state.xmem.read_address(
                DST_BASE_ROW * ROW_BYTES, self.dst_rows * ROW_BYTES
            )
            Path(self.output_path).write_bytes(bytes(data))
