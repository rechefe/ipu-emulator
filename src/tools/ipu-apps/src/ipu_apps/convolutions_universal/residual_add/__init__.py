"""Universal residual add: INT8 + INT8 -> INT32, flexible channel count.

Adds two INT8 tensors element-wise and stores the result as INT32.
Both tensors use the same paired-channel 128-byte chunk layout as the other
convolutions_universal apps (2 channels per 128-byte chunk, 64 bytes each for 8x8 spatial).

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
from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants ---------------------------------------------------------------

SPATIAL = 64  # 8x8

# -- Memory layout -----------------------------------------------------------

INPUT_A_BASE = 0x00000
INPUT_B_BASE = 0x80000
OUTPUT_BASE = 0x200000

ACC_CHUNK_BYTES = 512  # 128 lanes x 4 bytes


def _build_input_data(input_raw: bytes, num_channels: int) -> bytes:
    """Pack input into paired-chunk layout.

    Input: input_raw[ch * SPATIAL + pos] (per-channel, SPATIAL bytes each).

    Output: num_channels/2 chunks of 128 bytes.
      Chunk j: channel 2j in bytes 0-63, channel 2j+1 in bytes 64-127.
    """
    ch_pairs = num_channels // 2
    packed = bytearray(ch_pairs * 128)

    for j in range(ch_pairs):
        dst = j * 128
        src_even = (2 * j) * SPATIAL
        src_odd = (2 * j + 1) * SPATIAL
        packed[dst:dst + SPATIAL] = input_raw[src_even:src_even + SPATIAL]
        packed[dst + SPATIAL:dst + 128] = input_raw[src_odd:src_odd + SPATIAL]

    return bytes(packed)


class ResidualAddApp(IpuApp):
    """Universal residual add: INT8 + INT8 -> INT32, flexible channel count.

    Args:
        inst_path:    Path to assembled binary.
        input_a_path: Path to tensor A binary (packed paired-chunk format).
        input_b_path: Path to tensor B binary (packed paired-chunk format).
        output_path:  Optional path to write INT32 output.
        num_channels: Number of channels (must be even and >= 2).
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

        if num_channels < 2 or num_channels % 2 != 0:
            raise ValueError(f"num_channels must be even and >= 2, got {num_channels}")

        self.num_channels = num_channels
        self.ch_pairs = num_channels // 2
        self.total_input_bytes = self.ch_pairs * 128
        self.total_output_bytes = self.ch_pairs * ACC_CHUNK_BYTES

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(DType.INT8))

        input_a = self.input_a_path.read_bytes()
        state.xmem.write_address(INPUT_A_BASE, input_a)

        input_b = self.input_b_path.read_bytes()
        state.xmem.write_address(INPUT_B_BASE, input_b)

        state.regfile.set_cr(0, INPUT_A_BASE)
        state.regfile.set_cr(1, INPUT_B_BASE)
        state.regfile.set_cr(2, OUTPUT_BASE)
        state.regfile.set_cr(4, self.total_input_bytes)
        state.regfile.set_cr(5, 1)  # identity scalar for mult.ve.cr

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            result = state.xmem.read_address(OUTPUT_BASE, self.total_output_bytes)
            Path(self.output_path).write_bytes(bytes(result))
