"""Standard 3x3 convolution harness (96 input -> 96 output channels, 32x32).

32x32 spatial, 4 rows packed per 128-byte chunk (8 groups).
Each output channel has 96 input-channel 3x3 kernels (864 bytes total).
Since 864 > 128, each filter is stored as 12 x 128-byte blocks:
  block 0:  input channels 0-7   (72 bytes + 56 padding)
  block 1:  input channels 8-15  (72 bytes + 56 padding)
  ...
  block 11: input channels 88-95 (72 bytes + 56 padding)
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

ROWS = 32
COLS = 32
IN_CHANNELS = 96
OUT_CHANNELS = 96
KERNEL_H = 3
KERNEL_W = 3
KERNEL_SIZE = KERNEL_H * KERNEL_W  # 9
TAPS_PER_FILTER = IN_CHANNELS * KERNEL_SIZE  # 864

ROWS_PER_CHUNK = 128 // COLS  # 4
NUM_GROUPS = ROWS // ROWS_PER_CHUNK  # 8

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x020000
OUTPUT_BASE_ADDR = 0x050000
MASK_BASE_ADDR = 0x0B0000

# Each filter: 12 blocks x 128 bytes = 1536 bytes
CHANNELS_PER_BLOCK = 8
BLOCKS_PER_FILTER = IN_CHANNELS // CHANNELS_PER_BLOCK  # 12
FILTER_PADDED_BYTES = BLOCKS_PER_FILTER * 128  # 1536
KERNEL_TOTAL_BYTES = OUT_CHANNELS * FILTER_PADDED_BYTES  # 147456

OUTPUT_CHUNK_BYTES = 128 * 4  # 512 (128 elements x 4-byte acc)

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

    Layout (8 slots of 16 bytes = 128 bits each, 4 rows x 32 cols packed):
      slot 0: all zeros                 -- no masking (kc=0)
      slot 1: bits {0, 32, 64, 96}      -- zero col 0 of each row (kc=-1)
      slot 2: bits {31, 63, 95, 127}    -- zero col 31 of each row (kc=+1)
      slot 3: bits {96..127}            -- zero bottom row (last group kr=+1, kc=0)
      slot 4: bits {0, 32, 64, 96..127} -- left + bottom (last group kr=+1, kc=-1)
      slot 5: bits {31, 63, 95..127}    -- right + bottom (last group kr=+1, kc=+1)
      slots 6-7: unused (zeros)
    """
    mask = bytearray(128)

    def set_bit(slot: int, bit: int) -> None:
        byte_idx = slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    def set_bit_range(slot: int, lo: int, hi: int) -> None:
        for b in range(lo, hi + 1):
            set_bit(slot, b)

    # Slot 1: left border -- col 0 of each of 4 rows
    set_bit(1, 0)
    set_bit(1, 32)
    set_bit(1, 64)
    set_bit(1, 96)

    # Slot 2: right border -- col 31 of each of 4 rows
    set_bit(2, 31)
    set_bit(2, 63)
    set_bit(2, 95)
    set_bit(2, 127)

    # Slot 3: bottom row of chunk (row 3, positions 96-127)
    set_bit_range(3, 96, 127)

    # Slot 4: left border + bottom row
    set_bit(4, 0)
    set_bit(4, 32)
    set_bit(4, 64)
    set_bit_range(4, 96, 127)

    # Slot 5: right border + bottom row
    set_bit(5, 31)
    set_bit(5, 63)
    set_bit_range(5, 95, 127)

    return bytes(mask)


def _build_kernel_data(kernel_raw: bytes) -> bytes:
    """Pack kernel into 96 filters x 12 blocks x 128 bytes.

    Input: 82944 bytes (96 filters x 96 input channels x 9 taps, contiguous).
      kernel_raw[f * 864 + ic * 9 + tap]

    Output: 147456 bytes (96 filters x 12 blocks x 128 bytes).
      Block b: input channels b*8..(b+1)*8-1 (72 bytes + 56 padding)
    """
    kernel_padded = bytearray(KERNEL_TOTAL_BYTES)
    taps_per_half = CHANNELS_PER_BLOCK * KERNEL_SIZE  # 72 bytes per block
    for f in range(OUT_CHANNELS):
        for block in range(BLOCKS_PER_FILTER):
            src_offset = f * TAPS_PER_FILTER + block * taps_per_half
            dst_offset = (f * BLOCKS_PER_FILTER + block) * 128
            kernel_padded[dst_offset:dst_offset + taps_per_half] = (
                kernel_raw[src_offset:src_offset + taps_per_half]
            )
    return bytes(kernel_padded)


class Conv32x32x96App(IpuApp):
    """Standard 3x3 convolution application harness (96->96 channels, 32x32).

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input binary (8 groups x 96 ch x 128 bytes = 98304 bytes).
        kernel_path: Path to kernel binary (82944 bytes = 96 x 96 x 9 taps, raw).
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

        # Load input (8 groups x 96 channels x 128 bytes = 98304 bytes)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (82944 bytes raw -> padded to 147456 bytes)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = _build_kernel_data(kernel_raw)
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
            # Output: 8 groups x 96 output channels = 768 chunks of 512 bytes
            total_chunks = NUM_GROUPS * OUT_CHANNELS
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_chunks,
            )
