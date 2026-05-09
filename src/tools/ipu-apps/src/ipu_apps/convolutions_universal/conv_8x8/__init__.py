"""Universal standard 3x3 convolution harness: 8x8 spatial, flexible channels.

Paired-filter processing: two output filters share one accumulator.
Filter f0 (even) in lanes 0-63, filter f1 (odd) in lanes 64-127.

Input layout: 2 channels per 128-byte chunk (ch_even bytes 0-63, ch_odd 64-127).
Kernel layout: ceil(in_channels/8) blocks per filter, 128 bytes each.
  Each block: 8 channels x 9 taps = 72 bytes + 56 padding.
  Filters packed sequentially: f0 blocks, f1 blocks, f2 blocks, ...
Output: INT32 accumulator, 512 bytes per filter pair (128 lanes x 4 bytes).

Usage::

    from ipu_apps.convolutions_universal.conv_8x8 import Conv8x8App

    app = Conv8x8App(
        inst_path="conv_8x8.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        in_channels=64, out_channels=64,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import pack_input_paired, dump_outputs

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants ---------------------------------------------------------------

ROWS = 8
COLS = 8
SPATIAL = ROWS * COLS  # 64 bytes per channel

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x80000
MASK_BASE_ADDR = 0x200000
OUTPUT_BASE_ADDR = 0x200300

ACC_CHUNK_BYTES = 512  # 128 lanes x 4 bytes
CHANNELS_PER_BLOCK = 8
KERNEL_SIZE = 9


def _build_mask_data() -> bytes:
    """Build 512 bytes of mask data: 4 groups of 128 bytes each.

    Group A (offset 0, f0 channel A, cyclic base 128, active lanes 0-63):
      slot 0: bits {64-127}                              (kc=0)
      slot 1: bits {64-127} + {0,8,16,24,32,40,48,56}   (kc=-1)
      slot 2: bits {64-127} + {7,15,23,31,39,47,55,63}  (kc=+1)
      slot 3: bits {56-127}                              (kr=+1 bleed)
      slot 4: bits {56-127} + {0,8,16,24,32,40,48}      (kr=+1 bleed, kc=-1)
      slot 5: bits {56-127} + {7,15,23,31,39,47,55}     (kr=+1 bleed, kc=+1)

    Group B (offset 128, f0 channel B, cyclic base 192, active lanes 0-63):
      slot 0-2: same as group A
      slot 3: bits {0-7} + {64-127}                      (kr=-1 bleed)
      slot 4: bits {0-7} + {8,...,56} + {64-127}
      slot 5: bits {0-7} + {15,...,63} + {64-127}

    Group C (offset 256, f1 channel A, cyclic base 64, active lanes 64-127):
      slot 0: bits {0-63}
      slot 1: bits {0-63} + {64,72,...,120}
      slot 2: bits {0-63} + {71,79,...,127}
      slot 3: bits {0-63} + {120-127}
      slot 4: bits {0-63} + {120-127} + {64,72,...,112}
      slot 5: bits {0-63} + {120-127} + {71,79,...,119}

    Group D (offset 384, f1 channel B, cyclic base 128, active lanes 64-127):
      slot 0-2: same as group C
      slot 3: bits {0-71}
      slot 4: bits {0-71} + {72,80,...,120}
      slot 5: bits {0-71} + {79,87,...,127}
    """
    mask = bytearray(512)

    def set_bit(group: int, slot: int, bit: int) -> None:
        byte_idx = group * 128 + slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    def set_bit_range(group: int, slot: int, lo: int, hi: int) -> None:
        for b in range(lo, hi + 1):
            set_bit(group, slot, b)

    # Groups A,B (f0): zero lanes 64-127, active lanes 0-63
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

    # Groups C,D (f1): zero lanes 0-63, active lanes 64-127
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


def _build_kernel_data(
    kernel_raw: bytes,
    in_channels: int,
    out_channels: int,
) -> bytes:
    """Pack kernel into block format for sequential filter processing.

    Input: kernel_raw[oc * in_channels * 9 + ic * 9 + tap] for each filter,
           input channel, and tap (row-major 3x3).

    Output: Filters packed sequentially. Each filter has ceil(in_channels/8)
            blocks of 128 bytes. Within each block: 8 channels x 9 taps = 72 bytes,
            zero-padded to 128.
    """
    blocks_per_filter = math.ceil(in_channels / CHANNELS_PER_BLOCK)
    kernel_bytes_per_filter = blocks_per_filter * 128
    packed = bytearray(out_channels * kernel_bytes_per_filter)

    for oc in range(out_channels):
        for ic in range(in_channels):
            block = ic // CHANNELS_PER_BLOCK
            pos_in_block = ic % CHANNELS_PER_BLOCK
            for tap in range(KERNEL_SIZE):
                src = oc * in_channels * KERNEL_SIZE + ic * KERNEL_SIZE + tap
                dst = oc * kernel_bytes_per_filter + block * 128 + pos_in_block * KERNEL_SIZE + tap
                packed[dst] = kernel_raw[src]

    return bytes(packed)


# Back-compat alias: benchmarks and profiling scripts import this name
_build_input_data = pack_input_paired


class Conv8x8App(IpuApp):
    """Universal standard 3x3 convolution: 8x8 spatial, flexible channels.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input binary (packed paired-chunk format).
        kernel_path:  Path to kernel binary (sequential blocks per filter).
        output_path:  Optional path to write output.
        in_channels:  Number of input channels (must be even, >= 2).
        out_channels: Number of output channels (must be even, >= 2).
    """

    ASM_PATH = Path(__file__).resolve().parent / "conv_8x8.asm"

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)

        if in_channels < 2 or in_channels % 2 != 0:
            raise ValueError(f"in_channels must be even and >= 2, got {in_channels}")
        if out_channels < 2 or out_channels % 2 != 0:
            raise ValueError(f"out_channels must be even and >= 2, got {out_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ic_pairs = in_channels // 2
        self.oc_pairs = out_channels // 2
        self.blocks_per_filter = math.ceil(in_channels / CHANNELS_PER_BLOCK)
        self.kernel_bytes_per_filter = self.blocks_per_filter * 128
        self.total_input_bytes = self.ic_pairs * 128
        self.total_output_bytes = self.oc_pairs * ACC_CHUNK_BYTES

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(DType.INT8))

        # Load input (paired chunks)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (sequential blocks per filter)
        kernel_data = self.kernel_path.read_bytes()
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_data)

        # Load mask (4 groups x 128 bytes = 512 bytes)
        mask_data = _build_mask_data()
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)
        state.regfile.set_cr(4, self.kernel_bytes_per_filter)
        state.regfile.set_cr(5, self.total_input_bytes)
        state.regfile.set_cr(6, self.total_output_bytes)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_outputs(state, self.output_path, OUTPUT_BASE_ADDR, ACC_CHUNK_BYTES, self.oc_pairs)
