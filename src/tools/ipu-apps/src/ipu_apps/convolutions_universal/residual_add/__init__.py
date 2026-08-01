"""Universal residual add (wide-vector FP32): FP32 + FP32 -> FP32.

Adds two FP32 tensors element-wise and stores the result as FP32, running the
emulator in **wide-vector debug mode** (GitHub issue #33). In that mode every
lane is 4 bytes, so each channel occupies a full 512-byte chunk of 128 FP32
lanes and ``STR_ACC_REG`` writes the full 512-byte accumulator.

Layout (per tensor): ``num_channels`` consecutive 512-byte chunks, one channel
per chunk, 128 little-endian FP32 lanes each.

MobileViT S residual stages: 64x64x64, 32x32x96, 16x16x128, 8x8x160.

Usage::

    from ipu_apps.convolutions_universal.residual_add import ResidualAddApp

    app = ResidualAddApp(
        inst_path="residual_add.bin",
        input_a_path="tensor_a.bin",
        input_b_path="tensor_b.bin",
        output_path="output.bin",
        num_channels=160,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_apps.base import IpuApp

# -- Constants ---------------------------------------------------------------

LANES_PER_CHUNK = 128
WIDE_CHUNK_BYTES = LANES_PER_CHUNK * 4  # 512: one channel, 128 FP32 lanes

# -- Memory layout -----------------------------------------------------------

INPUT_A_BASE = 0x00000
INPUT_B_BASE = 0x80000
OUTPUT_BASE = 0x100000


class ResidualAddApp(IpuApp):
    """Universal residual add: FP32 + FP32 -> FP32 (wide-vector debug mode).

    Args:
        inst_path:    Path to assembled binary.
        input_a_path: Path to tensor A binary (512-byte FP32 chunks).
        input_b_path: Path to tensor B binary (512-byte FP32 chunks).
        output_path:  Optional path to write FP32 output.
        num_channels: Number of channels (>= 1).
    """

    ASM_PATH = Path(__file__).resolve().parent / "residual_add.asm"

    def __init__(
        self,
        *,
        num_channels: int,
        input_a_path: str | Path,
        input_b_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_a_path = Path(input_a_path)
        self.input_b_path = Path(input_b_path)

        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")

        self.num_channels = num_channels
        self.total_input_bytes = num_channels * WIDE_CHUNK_BYTES
        self.total_output_bytes = num_channels * WIDE_CHUNK_BYTES
        self.output_base = OUTPUT_BASE

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires."""
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
        )
        # dtype is unused on the FP32 wide path, but several helpers branch on
        # it; INT8 matches the existing wide-vector tests.
        state.dtype = DType.INT8
        return state

    def setup(self, state: "IpuState") -> None:

        input_a = self.input_a_path.read_bytes()
        state.xmem.write_address(INPUT_A_BASE, input_a)

        input_b = self.input_b_path.read_bytes()
        state.xmem.write_address(INPUT_B_BASE, input_b)

        # CR0/CR1 are reserved config registers (read as 0 and 1); the asm
        # reuses cr0 as a zero source and cr1 as the identity scalar.
        state.regfile.set_cr(2, INPUT_A_BASE)
        state.regfile.set_cr(3, INPUT_B_BASE)
        state.regfile.set_cr(4, OUTPUT_BASE)
        state.regfile.set_cr(5, WIDE_CHUNK_BYTES)  # chunk step (512)
        state.regfile.set_cr(6, self.total_input_bytes)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            result = state.xmem.read_address(OUTPUT_BASE, self.total_output_bytes)
            Path(self.output_path).write_bytes(bytes(result))

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)
