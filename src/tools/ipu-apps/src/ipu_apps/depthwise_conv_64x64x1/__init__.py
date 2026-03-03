"""Depthwise convolution test harness (1 channel, 64x64, 3x3 kernel).

64x64 spatial, 1 channel.  Since spatial width (64) < SIMD width (128),
2 spatial rows are packed per 128-byte chunk (32 chunks total).

Usage::

    from ipu_apps.depthwise_conv_64x64x1 import DepthwiseConv64x64x1App

    app = DepthwiseConv64x64x1App(
        inst_path="depthwise_conv_64x64x1.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        dtype="INT8",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants ---------------------------------------------------------------

ROWS = 64
COLS = 64
KERNEL_SIZE = 9  # 3x3

ROWS_PER_CHUNK = 128 // COLS  # 2
NUM_CHUNKS = ROWS // ROWS_PER_CHUNK  # 32

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x20000
MASK_BASE_ADDR = 0x30000
OUTPUT_BASE_ADDR = 0x40000

# Output: 32 chunks x 512 bytes (128 elements x 4-byte accumulator words)
OUTPUT_ROW_BYTES = 128 * 4  # 512

_DTYPE_MAP = {
    "INT8": DType.INT8,
    "int8": DType.INT8,
    "FP8_E4M3": DType.FP8_E4M3,
    "fp8_e4m3": DType.FP8_E4M3,
    "FP8_E5M2": DType.FP8_E5M2,
    "fp8_e5m2": DType.FP8_E5M2,
}


def parse_dtype(dtype_str: str) -> DType:
    """Parse a dtype string into a :class:`DType` enum value."""
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(
            f"Invalid dtype '{dtype_str}'. Supported: INT8, FP8_E4M3, FP8_E5M2"
        )
    return dt


def _build_mask_data() -> bytes:
    """Build the 128-byte mask register data.

    Layout (8 slots of 16 bytes = 128 bits each):
      slot 0: all zeros             -- no masking (kc=0)
      slot 1: bits {0, 64} set      -- zero col 0 of each packed row (kc=-1)
      slot 2: bits {63, 127} set    -- zero col 63 of each packed row (kc=+1)
      slot 3: bits {64..127}        -- zero bottom row (last chunk kr=+1, kc=0)
      slot 4: bits {0, 64..127}     -- left border + bottom row (last chunk kr=+1, kc=-1)
      slot 5: bits {63..127}        -- right border + bottom row (last chunk kr=+1, kc=+1)
      slots 6-7: unused (zeros)
    """
    mask = bytearray(128)

    def set_bit(slot: int, bit: int) -> None:
        byte_idx = slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    def set_bit_range(slot: int, lo: int, hi: int) -> None:
        for b in range(lo, hi + 1):
            set_bit(slot, b)

    # Slot 1: bits 0, 64
    set_bit(1, 0)
    set_bit(1, 64)

    # Slot 2: bits 63, 127
    set_bit(2, 63)
    set_bit(2, 127)

    # Slot 3: bits 64..127
    set_bit_range(3, 64, 127)

    # Slot 4: bits {0} + {64..127}
    set_bit(4, 0)
    set_bit_range(4, 64, 127)

    # Slot 5: bits {63..127}
    set_bit_range(5, 63, 127)

    return bytes(mask)


class DepthwiseConv64x64x1App(IpuApp):
    """Depthwise convolution application harness -- 64x64x1 variant.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input image binary (64x64 = 4096 bytes).
        kernel_path: Path to kernel binary (9 bytes, padded to 128).
        output_path: Optional path to write output.
        dtype:       Data type string or :class:`DType`.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        # Set data type
        state.set_cr_dtype(int(self.dtype))

        # Load input image (32 chunks x 128 bytes = 4096 bytes)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (padded to 128 bytes, only first 9 matter)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = bytearray(128)
        kernel_padded[:len(kernel_raw)] = kernel_raw
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_padded)

        # Load mask data
        mask_data = _build_mask_data()
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # Set CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_ROW_BYTES, NUM_CHUNKS,
            )
