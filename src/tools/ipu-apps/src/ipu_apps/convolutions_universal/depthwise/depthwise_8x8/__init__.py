"""Universal depthwise 3x3 convolution harness: 8x8 spatial, flexible channels.

Each channel has its own independent 3x3 kernel (no cross-channel mixing).
Paired-channel processing: channels 2k and 2k+1 share one accumulator.
Channel A (even) in lanes 0-63, channel B (odd) in lanes 64-127.

Input layout: 2 channels per 128-byte chunk (ch_even bytes 0-63, ch_odd 64-127).
Kernel layout: ceil(channels/8) groups of 128 bytes (8 ch x 9 taps + padding).
Output: 128 bytes int8 per channel pair (AAQ-quantized).

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_8x8 import Depthwise8x8App

    app = Depthwise8x8App(
        inst_path="depthwise_8x8.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        channels=160,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    parse_dtype,
    pack_input_paired,
    dump_outputs,
)

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

CHANNELS_PER_GROUP = 8
KERNEL_SIZE = 9
OUTPUT_CHUNK_BYTES = 128  # 128 bytes int8 per channel pair (AAQ-quantized)
ACC_CHUNK_BYTES = OUTPUT_CHUNK_BYTES  # alias for backward compat

# Back-compat alias: benchmarks and profiling scripts import this name
_build_input_data = pack_input_paired


def _build_mask_data() -> bytes:
    """Build 512 bytes of mask data: 4 groups of 128 bytes each.

    Only groups A (offset 0) and D (offset 384) are used:

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

    # Group A,B (f0): zero lanes 64-127, active lanes 0-63
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


def _build_kernel_data(kernel_raw: bytes, num_channels: int) -> bytes:
    """Pack kernel into grouped format for paired processing.

    Input: kernel_raw[ch * 9 + tap] for each channel and tap (row-major 3x3).

    Output: ceil(num_channels/8) groups of 128 bytes.
      Within each group: 8 channels x 9 bytes = 72 bytes, zero-padded to 128.
    """
    num_groups = math.ceil(num_channels / CHANNELS_PER_GROUP)
    packed = bytearray(num_groups * 128)

    for ch in range(num_channels):
        group = ch // CHANNELS_PER_GROUP
        pos_in_group = ch % CHANNELS_PER_GROUP
        for tap in range(KERNEL_SIZE):
            dst = group * 128 + pos_in_group * KERNEL_SIZE + tap
            packed[dst] = kernel_raw[ch * KERNEL_SIZE + tap]

    return bytes(packed)


class Depthwise8x8App(IpuApp):
    """Universal depthwise 3x3 convolution: 8x8 spatial, flexible channels.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input binary (packed paired-chunk format).
        kernel_path:  Path to kernel binary (grouped 128-byte blocks).
        output_path:  Optional path to write output.
        channels:     Number of channels (must be multiple of 8, >= 8).
        dtype:        Data type string or :class:`DType` (default ``"INT8"``).
    """

    ASM_PATH = Path(__file__).resolve().parent / "depthwise_8x8.asm"

    def __init__(
        self,
        *,
        channels: int | None = None,
        dtype: str | DType = "INT8",
        **kwargs,
    ) -> None:
        # Back-compat: accept legacy kwarg num_channels
        if channels is None:
            if "num_channels" in kwargs:
                warnings.warn(
                    "Depthwise8x8App: 'num_channels' is deprecated, use 'channels'",
                    DeprecationWarning,
                    stacklevel=2,
                )
                channels = kwargs.pop("num_channels")
            else:
                raise TypeError(
                    "Depthwise8x8App() missing required keyword argument: 'channels'"
                )

        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)
        self.dtype = parse_dtype(dtype)

        if channels < 8 or channels % 8 != 0:
            raise ValueError(
                f"channels must be a multiple of 8 and >= 8, got {channels}"
            )

        self.channels = channels
        self.num_channels = channels  # back-compat alias
        self.num_pairs = channels // 2
        self.num_kernel_groups = math.ceil(channels / CHANNELS_PER_GROUP)
        self.total_input_bytes = self.num_pairs * 128
        self.total_output_bytes = self.num_pairs * OUTPUT_CHUNK_BYTES

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))

        # Load input (paired chunks)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (grouped blocks)
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
        state.regfile.set_cr(4, self.total_input_bytes)
        state.regfile.set_cr(5, 0)      # zero constant
        state.regfile.set_cr(6, 119)    # cyclic base offset for 3x3 tap start (kr=-1, kc=-1)
        state.regfile.set_cr(12, 128)   # step constant: chunk / kernel group / output advance
        state.regfile.set_cr(13, 72)    # pair sub-loop end (4 pairs x 18 bytes)
        state.regfile.set_cr(14, 384)   # mask group D offset

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_outputs(state, self.output_path, OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, self.num_pairs)
