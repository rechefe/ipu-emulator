"""Standard 3x3 convolution harness (16 input -> 16 output channels, 8x8).

Paired-filter processing: two output filters share one accumulator.
Filter f0 (even) accumulates in lanes 0-63, filter f1 (odd) in lanes 64-127.
One str_acc_reg per pair stores all 512 bytes (both filters valid).

8x8 spatial with mask-swapping technique: 2 input channels are packed per
128-byte chunk (64 bytes each).

Four mask groups (128 bytes each) are stored in XMEM:
  Groups A,B: for f0 (zero lanes 64-127, active lanes 0-63)
  Groups C,D: for f1 (zero lanes 0-63, active lanes 64-127)

Kernel per filter: 16 * 9 = 144 bytes -> 2 r0 blocks of 128 bytes each.
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
IN_CHANNELS = 16
OUT_CHANNELS = 16
NUM_PAIRS = OUT_CHANNELS // 2  # 8

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x00800
MASK_BASE_ADDR = 0x01800
OUTPUT_BASE_ADDR = 0x02000

# Input: 8 packed chunks x 128 bytes = 1024 bytes
NUM_CHUNKS = IN_CHANNELS // 2  # 8

# Kernel: 2 blocks per filter, 128 bytes each
BLOCKS_PER_FILTER = 2
CHANNELS_PER_BLOCK = 8
TAPS_PER_CHANNEL = 9
FILTER_BLOCK_BYTES = 128
FILTER_PADDED_BYTES = BLOCKS_PER_FILTER * FILTER_BLOCK_BYTES  # 256
KERNEL_TOTAL_BYTES = OUT_CHANNELS * FILTER_PADDED_BYTES  # 4096

# Output: str_acc_reg stores 128 elements x 4 bytes = 512 bytes per pair
# All 128 lanes are valid (f0 in 0-63, f1 in 64-127).
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

    Groups A,B (f0, active lanes 0-63, zero lanes 64-127):

      Group A (for channel A, cyclic base 128):
        slot 0: bits {64-127}                              (kc=0, normal)
        slot 1: bits {64-127} + {0,8,16,24,32,40,48,56}   (kc=-1, left)
        slot 2: bits {64-127} + {7,15,23,31,39,47,55,63}  (kc=+1, right)
        slot 3: bits {56-127}                              (kr=+1 bleed, kc=0)
        slot 4: bits {56-127} + {0,8,16,24,32,40,48}      (kr=+1 bleed, kc=-1)
        slot 5: bits {56-127} + {7,15,23,31,39,47,55}     (kr=+1 bleed, kc=+1)

      Group B (for channel B, cyclic base 192):
        slot 0-2: identical to group A slots 0-2
        slot 3: bits {0-7} + {64-127}                     (kr=-1 bleed, kc=0)
        slot 4: bits {0-7} + {8,16,24,32,40,48,56} + {64-127}
        slot 5: bits {0-7} + {15,23,31,39,47,55,63} + {64-127}

    Groups C,D (f1, active lanes 64-127, zero lanes 0-63):

      Group C (for channel A, cyclic base 64):
        slot 0: bits {0-63}                                    (kc=0, normal)
        slot 1: bits {0-63} + {64,72,80,88,96,104,112,120}    (kc=-1, left)
        slot 2: bits {0-63} + {71,79,87,95,103,111,119,127}   (kc=+1, right)
        slot 3: bits {0-63} + {120-127}                        (kr=+1 bleed, kc=0)
        slot 4: bits {0-63} + {120-127} + {64,72,80,88,96,104,112}
        slot 5: bits {0-63} + {120-127} + {71,79,87,95,103,111,119}

      Group D (for channel B, cyclic base 128):
        slot 0-2: identical to group C slots 0-2
        slot 3: bits {0-71}                                    (kr=-1 bleed, kc=0)
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

    # ---- Shared slots 0-2 (identical for groups A and B) ----
    for g in range(2):
        set_bit_range(g, 0, 64, 127)

        set_bit_range(g, 1, 64, 127)
        for row in range(8):
            set_bit(g, 1, row * 8)  # col 0

        set_bit_range(g, 2, 64, 127)
        for row in range(8):
            set_bit(g, 2, row * 8 + 7)  # col 7

    # ---- Group A slots 3-5 (kr=+1 bottom bleed: zero lanes 56-63) ----
    set_bit_range(0, 3, 56, 127)

    set_bit_range(0, 4, 56, 127)
    for row in range(7):
        set_bit(0, 4, row * 8)

    set_bit_range(0, 5, 56, 127)
    for row in range(7):
        set_bit(0, 5, row * 8 + 7)

    # ---- Group B slots 3-5 (kr=-1 top bleed: zero lanes 0-7) ----
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

    # ---- Shared slots 0-2 (identical for groups C and D) ----
    for g in (2, 3):
        set_bit_range(g, 0, 0, 63)

        set_bit_range(g, 1, 0, 63)
        for row in range(8):
            set_bit(g, 1, 64 + row * 8)  # col 0 in 64-127 space

        set_bit_range(g, 2, 0, 63)
        for row in range(8):
            set_bit(g, 2, 64 + row * 8 + 7)  # col 7 in 64-127 space

    # ---- Group C slots 3-5 (kr=+1 bottom bleed: zero lanes 120-127) ----
    set_bit_range(2, 3, 0, 63)
    set_bit_range(2, 3, 120, 127)

    set_bit_range(2, 4, 0, 63)
    set_bit_range(2, 4, 120, 127)
    for row in range(7):  # rows 0-6 in 64-127 space (row 7 already in 120-127)
        set_bit(2, 4, 64 + row * 8)

    set_bit_range(2, 5, 0, 63)
    set_bit_range(2, 5, 120, 127)
    for row in range(7):
        set_bit(2, 5, 64 + row * 8 + 7)

    # ---- Group D slots 3-5 (kr=-1 top bleed: zero lanes 64-71) ----
    set_bit_range(3, 3, 0, 71)

    set_bit_range(3, 4, 0, 71)
    for row in range(1, 8):  # rows 1-7 (row 0 already in 64-71)
        set_bit(3, 4, 64 + row * 8)

    set_bit_range(3, 5, 0, 71)
    for row in range(1, 8):
        set_bit(3, 5, 64 + row * 8 + 7)

    return bytes(mask)


class Conv8x8_16to16App(IpuApp):
    """Standard 3x3 convolution application harness (16->16 channels, 8x8).

    Paired-filter processing: filters are processed in pairs, with f0 (even)
    in lanes 0-63 and f1 (odd) in lanes 64-127 of the same accumulator.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input binary (1024 bytes: 8 chunks x 128 bytes).
        kernel_path: Path to kernel binary (4096 bytes: 16 filters x 256 bytes).
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

        # Load input (8 chunks x 128 bytes = 1024 bytes)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (16 filters x 2 blocks x 128 bytes = 4096 bytes)
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
            # Each pair stores 512 bytes: f0 (lanes 0-63, 256B) + f1 (lanes 64-127, 256B).
            # All 128 lanes are valid.
            result = bytearray()
            for pair in range(NUM_PAIRS):
                addr = OUTPUT_BASE_ADDR + pair * ACC_CHUNK_BYTES
                chunk = state.xmem.read_address(addr, ACC_CHUNK_BYTES)
                result.extend(chunk)
            Path(self.output_path).write_bytes(bytes(result))
