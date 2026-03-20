"""Depthwise 3x3 convolution harness (160 channels, 8x8 spatial).

Each channel has its own independent 3x3 kernel (no cross-channel mixing).
Paired-channel processing: channels 2k and 2k+1 share one accumulator.
Channel A (even) in lanes 0-63, channel B (odd) in lanes 64-127.
One str_acc_reg per pair stores all 512 bytes (both channels valid).

8x8 spatial with packed pairs: 2 input channels per 128-byte chunk.

Mask groups (same structure as standard conv 8x8):
  Group A (offset 0):   for channel A (zero lanes 64-127, bleed for kr=+1)
  Group D (offset 384): for channel B (zero lanes 0-63, bleed for kr=-1)

Kernel: 160 channels x 9 bytes -> 20 groups of 128 bytes each.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants ---------------------------------------------------------------

ROWS = 8
COLS = 8
NUM_CHANNELS = 160
NUM_PAIRS = NUM_CHANNELS // 2  # 80

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x02800
MASK_BASE_ADDR = 0x03200
OUTPUT_BASE_ADDR = 0x03400

# Input: 80 packed chunks x 128 bytes = 10240 bytes
NUM_CHUNKS = NUM_CHANNELS // 2  # 80

# Kernel: 20 groups of 128 bytes (8 channels x 9 bytes + padding)
CHANNELS_PER_GROUP = 8
KERNEL_SIZE = 9
NUM_KERNEL_GROUPS = NUM_CHANNELS // CHANNELS_PER_GROUP  # 20
KERNEL_GROUP_BYTES = 128
KERNEL_TOTAL_BYTES = NUM_KERNEL_GROUPS * KERNEL_GROUP_BYTES  # 2560

# Output: 128 elements x 4 bytes = 512 bytes per pair
ACC_CHUNK_BYTES = 512

_DTYPE_MAP = {
    "INT8": DType.INT8,
    "int8": DType.INT8,
    "FP8_E4M3": DType.FP8_E4M3,
    "fp8_e4m3": DType.FP8_E4M3,
    "FP8_E5M2": DType.FP8_E5M2,
    "fp8_e5m2": DType.FP8_E5M2,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(
            f"Invalid dtype '{dtype_str}'. Supported: INT8, FP8_E4M3, FP8_E5M2"
        )
    return dt


def _build_mask_data() -> bytes:
    """Build 512 bytes of mask data: 4 groups of 128 bytes each.

    Only groups A and D are used for depthwise, but all 4 are built
    for compatibility with the standard conv mask layout.

    Group A (channel A, zero lanes 64-127, bleed for kr=+1):
      slot 0: bits {64-127}
      slot 1: bits {64-127} + {0,8,16,24,32,40,48,56}
      slot 2: bits {64-127} + {7,15,23,31,39,47,55,63}
      slot 3: bits {56-127}
      slot 4: bits {56-127} + {0,8,16,24,32,40,48}
      slot 5: bits {56-127} + {7,15,23,31,39,47,55}

    Group D (channel B, zero lanes 0-63, bleed for kr=-1):
      slot 0: bits {0-63}
      slot 1: bits {0-63} + {64,72,80,88,96,104,112,120}
      slot 2: bits {0-63} + {71,79,87,95,103,111,119,127}
      slot 3: bits {0-71}
      slot 4: bits {0-71} + {72,80,88,96,104,112,120}
      slot 5: bits {0-71} + {79,87,95,103,111,119,127}
    """
    mask = bytearray(512)

    def set_bit(group: int, slot: int, bit: int) -> None:
        byte_idx = group * 128 + slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    def set_bit_range(group: int, slot: int, lo: int, hi: int) -> None:
        for b in range(lo, hi + 1):
            set_bit(group, slot, b)

    # ========================================================================
    # Groups A,B (f0): zero lanes 64-127, active lanes 0-63
    # ========================================================================

    for g in range(2):
        set_bit_range(g, 0, 64, 127)

        set_bit_range(g, 1, 64, 127)
        for row in range(8):
            set_bit(g, 1, row * 8)

        set_bit_range(g, 2, 64, 127)
        for row in range(8):
            set_bit(g, 2, row * 8 + 7)

    # Group A slots 3-5 (kr=+1 bottom bleed)
    set_bit_range(0, 3, 56, 127)

    set_bit_range(0, 4, 56, 127)
    for row in range(7):
        set_bit(0, 4, row * 8)

    set_bit_range(0, 5, 56, 127)
    for row in range(7):
        set_bit(0, 5, row * 8 + 7)

    # Group B slots 3-5 (kr=-1 top bleed)
    set_bit_range(1, 3, 0, 7)
    set_bit_range(1, 3, 64, 127)

    set_bit_range(1, 4, 0, 7)
    set_bit_range(1, 4, 64, 127)
    for row in range(1, 8):
        set_bit(1, 4, row * 8)

    set_bit_range(1, 5, 0, 7)
    set_bit_range(1, 5, 64, 127)
    for row in range(1, 8):
        set_bit(1, 5, row * 8 + 7)

    # ========================================================================
    # Groups C,D (f1): zero lanes 0-63, active lanes 64-127
    # ========================================================================

    for g in (2, 3):
        set_bit_range(g, 0, 0, 63)

        set_bit_range(g, 1, 0, 63)
        for row in range(8):
            set_bit(g, 1, 64 + row * 8)

        set_bit_range(g, 2, 0, 63)
        for row in range(8):
            set_bit(g, 2, 64 + row * 8 + 7)

    # Group C slots 3-5 (kr=+1 bottom bleed)
    set_bit_range(2, 3, 0, 63)
    set_bit_range(2, 3, 120, 127)

    set_bit_range(2, 4, 0, 63)
    set_bit_range(2, 4, 120, 127)
    for row in range(7):
        set_bit(2, 4, 64 + row * 8)

    set_bit_range(2, 5, 0, 63)
    set_bit_range(2, 5, 120, 127)
    for row in range(7):
        set_bit(2, 5, 64 + row * 8 + 7)

    # Group D slots 3-5 (kr=-1 top bleed)
    set_bit_range(3, 3, 0, 71)

    set_bit_range(3, 4, 0, 71)
    for row in range(1, 8):
        set_bit(3, 4, 64 + row * 8)

    set_bit_range(3, 5, 0, 71)
    for row in range(1, 8):
        set_bit(3, 5, 64 + row * 8 + 7)

    return bytes(mask)


class DepthwiseConv8x8x160App(IpuApp):
    """Depthwise 3x3 convolution application harness (160 channels, 8x8).

    Paired-channel processing: channels are processed in pairs, with
    channel A (even) in lanes 0-63 and channel B (odd) in lanes 64-127.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input binary (10240 bytes: 80 chunks x 128 bytes).
        kernel_path: Path to kernel binary (2560 bytes: 20 groups x 128 bytes).
        output_path: Optional path to write output.
        dtype:       Data type string or :class:`DType`.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))

        # Load input (80 chunks x 128 bytes = 10240 bytes)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (20 groups x 128 bytes = 2560 bytes)
        kernel_data = self.kernel_path.read_bytes()
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_data)

        # Load mask data (4 groups x 128 bytes = 512 bytes)
        mask_data = _build_mask_data()
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # Set CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            result = bytearray()
            for pair in range(NUM_PAIRS):
                addr = OUTPUT_BASE_ADDR + pair * ACC_CHUNK_BYTES
                chunk = state.xmem.read_address(addr, ACC_CHUNK_BYTES)
                result.extend(chunk)
            Path(self.output_path).write_bytes(bytes(result))
