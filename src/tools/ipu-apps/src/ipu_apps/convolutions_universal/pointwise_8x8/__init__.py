"""Universal pointwise (1x1) convolution harness: 8x8 spatial, flexible channels.

Paired-output processing: two output channels share one accumulator.
OC f0 (even) in lanes 0-63, OC f1 (odd) in lanes 64-127.

Input layout: 2 channels per 128-byte chunk (IC_even bytes 0-63, IC_odd 64-127).
Kernel layout: interleaved per IC pair, packed into 128-byte blocks.
Output: 128 bytes int8 per OC pair (AAQ-quantized).

Usage::

    from ipu_apps.convolutions_universal.pointwise_8x8 import Pointwise8x8App

    app = Pointwise8x8App(
        inst_path="pointwise_8x8.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        in_channels=160, out_channels=160,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
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
OUTPUT_BASE_ADDR = 0x200100

OUTPUT_CHUNK_BYTES = 128  # 128 bytes int8 per OC pair (AAQ-quantized)


def _build_mask_data() -> bytes:
    """Build 128 bytes of mask data: 2 slots.

    Slot 0 (bytes 0-15):  bits {64-127} set -> zero lanes 64-127 (for f0)
    Slot 1 (bytes 16-31): bits {0-63} set  -> zero lanes 0-63   (for f1)
    Slots 2-7: zeros (unused).
    """
    mask = bytearray(128)

    def set_bit(slot: int, bit: int) -> None:
        byte_idx = slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    for b in range(64, 128):
        set_bit(0, b)
    for b in range(0, 64):
        set_bit(1, b)

    return bytes(mask)


def _build_kernel_data(
    kernel_raw: bytes,
    in_channels: int,
    out_channels: int,
) -> bytes:
    """Pack kernel into interleaved block format for paired processing.

    Input: kernel_raw[oc * in_channels + ic] for each output/input channel.

    Output: For each OC pair (f0=2p, f1=2p+1), blocks of 128 bytes.
      Per IC pair j (ICs 2j, 2j+1): 4 bytes = [f0[2j], f0[2j+1], f1[2j], f1[2j+1]]
      32 IC pairs per block. Last block zero-padded if partial.
    """
    ic_pairs = in_channels // 2
    oc_pairs = out_channels // 2
    blocks_per_pair = math.ceil(ic_pairs / 32)
    kernel_bytes_per_pair = blocks_per_pair * 128

    packed = bytearray(oc_pairs * kernel_bytes_per_pair)

    for p in range(oc_pairs):
        f0 = 2 * p
        f1 = 2 * p + 1
        for j in range(ic_pairs):
            block = j // 32
            pos_in_block = j % 32
            dst = p * kernel_bytes_per_pair + block * 128 + pos_in_block * 4
            ic_even = 2 * j
            ic_odd = 2 * j + 1
            packed[dst] = kernel_raw[f0 * in_channels + ic_even]
            packed[dst + 1] = kernel_raw[f0 * in_channels + ic_odd]
            packed[dst + 2] = kernel_raw[f1 * in_channels + ic_even]
            packed[dst + 3] = kernel_raw[f1 * in_channels + ic_odd]

    return bytes(packed)


# Back-compat alias: benchmarks and profiling scripts import this name
_build_input_data = pack_input_paired


class Pointwise8x8App(IpuApp):
    """Universal pointwise (1x1) convolution: 8x8 spatial, flexible channels.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input binary (packed paired-chunk format).
        kernel_path:  Path to kernel binary (packed interleaved block format).
        output_path:  Optional path to write output.
        in_channels:  Number of input channels (must be even, >= 2).
        out_channels: Number of output channels (must be even, >= 2).
    """

    ASM_PATH = Path(__file__).resolve().parent / "pointwise_8x8.asm"

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        dtype: str | DType = "INT8",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)
        self.dtype = parse_dtype(dtype)

        if in_channels < 2 or in_channels % 2 != 0:
            raise ValueError(f"in_channels must be even and >= 2, got {in_channels}")
        if out_channels < 2 or out_channels % 2 != 0:
            raise ValueError(f"out_channels must be even and >= 2, got {out_channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ic_pairs = in_channels // 2
        self.oc_pairs = out_channels // 2
        self.blocks_per_pair = math.ceil(self.ic_pairs / 32)
        self.kernel_bytes_per_pair = self.blocks_per_pair * 128
        self.total_input_bytes = self.ic_pairs * 128
        self.total_output_bytes = self.oc_pairs * OUTPUT_CHUNK_BYTES

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))

        # Load input (packed paired chunks)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (packed interleaved blocks)
        kernel_data = self.kernel_path.read_bytes()
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_data)

        # Load mask
        mask_data = _build_mask_data()
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)
        state.regfile.set_cr(4, self.kernel_bytes_per_pair)
        state.regfile.set_cr(5, self.total_input_bytes)
        state.regfile.set_cr(6, self.total_output_bytes)
        state.regfile.set_cr(7, 0)      # zero constant
        state.regfile.set_cr(12, 128)   # step constant: chunk / kernel block / output advance
        state.regfile.set_cr(13, 64)    # cyclic_offset for f1 x IC_even
        state.regfile.set_cr(14, 192)   # cyclic_offset for f0 x IC_odd

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_outputs(state, self.output_path, OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, self.oc_pairs)
