"""Universal depthwise 3x3 convolution harness.

A single parameterized harness that works for ANY valid depthwise convolution
configuration (spatial >= 16x16, channels >= 1).

Mirrors the standard ConvUniversal harness: FPB=28 super-block packing
(28 channels per 256-byte block), walking-pointer asm, rotating cyclic slots.

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
        DepthwiseConvUniversalApp,
    )

    app = DepthwiseConvUniversalApp(
        inst_path="depthwise_conv_universal.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        dtype="INT8",
        rows=64, cols=64, channels=256,
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
    build_border_mask_data,
    dump_outputs,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x110000
MASK_BASE_ADDR = 0x120000
ZERO_BASE_ADDR = 0x120080  # 128 bytes of zeros (right after mask data)
OUTPUT_BASE_ADDR = 0x130000

OUTPUT_CHUNK_BYTES = 128  # 128 bytes per output channel per chunk (int8)

FPB = 28  # channels per 256-byte super-block (R0+R1, shared fixed_idx)
SUPER_BLOCK_BYTES = 256


def _pack_depthwise_kernel_fpb28(kernel_raw: bytes, channels: int) -> bytes:
    """Pack per-channel weights into FPB=28 super-blocks.

    Input:  channels * 9 bytes — channel ch's 9 weights at offset ch*9.
    Output: ceil(channels/28) super-blocks of 256 bytes each.

    Within one super-block, channel s (s in 0..27) occupies bytes
    [s*9 .. s*9+9).  R0 holds bytes 0..127, R1 holds 128..255; mult.ve.cyclic
    with shared fixed_idx 0..255 transparently spans both halves (the
    straddler at byte 126..134 is half-R0/half-R1).
    """
    num_blocks = math.ceil(channels / FPB)
    total = num_blocks * SUPER_BLOCK_BYTES
    packed = bytearray(total)
    for sb in range(num_blocks):
        sb_base = sb * SUPER_BLOCK_BYTES
        for s in range(FPB):
            ch = sb * FPB + s
            if ch >= channels:
                break
            src = ch * 9
            dst = sb_base + s * 9
            packed[dst:dst + 9] = kernel_raw[src:src + 9]
    return bytes(packed)


class DepthwiseConvUniversalApp(IpuApp):
    """Universal depthwise 3x3 convolution application harness.

    Args:
        inst_path:    Path to assembled universal binary.
        input_path:   Path to input image binary.
        kernel_path:  Path to kernel binary (channels * 9 bytes, INT8 raw).
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height.
        cols:         Spatial width (16, 32, or 64).
        channels:     Number of channels (>= 1).
    """

    def __init__(
        self,
        *,
        dtype: str | DType = "INT8",
        rows: int,
        cols: int,
        channels: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

        valid_cols = {16, 32, 64}
        if cols not in valid_cols:
            raise ValueError(f"cols must be in {valid_cols}, got {cols}")
        num_chunks = (rows * cols) // 128
        if num_chunks < 2:
            raise ValueError(
                f"Need at least 2 chunks (rows*cols >= 256), got {rows}*{cols}={rows*cols}"
            )
        if channels < 1:
            raise ValueError(f"channels ({channels}) must be >= 1")

        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.num_chunks = num_chunks
        self.group_stride = channels * 128
        self.num_super_blocks = math.ceil(channels / FPB)
        self.total_kernel_bytes = self.num_super_blocks * SUPER_BLOCK_BYTES

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))

        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        kernel_raw = self.kernel_path.read_bytes()
        expected = self.channels * 9
        if len(kernel_raw) != expected:
            raise ValueError(
                f"kernel_path file has {len(kernel_raw)} bytes, "
                f"expected {expected} (channels * 9)"
            )
        kernel_packed = _pack_depthwise_kernel_fpb28(kernel_raw, self.channels)
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_packed)

        mask_data = build_border_mask_data(self.cols)
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        state.xmem.write_address(ZERO_BASE_ADDR, bytes(128))

        # Base-address CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)

        # Parameter CR registers
        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(5, self.num_chunks)
        state.regfile.set_cr(6, self.group_stride)
        state.regfile.set_cr(7, FPB * 128)        # 28 * 128 = 3584
        state.regfile.set_cr(8, self.total_kernel_bytes)
        state.regfile.set_cr(9, ZERO_BASE_ADDR)
        state.regfile.set_cr(10, 128)             # output-pointer step (128 B int8)
        state.regfile.set_cr(11, (self.num_chunks - 1) * self.group_stride)
        state.regfile.set_cr(12, 128)
        state.regfile.set_cr(13, 256)
        state.regfile.set_cr(14, (256 - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.channels
            dump_outputs(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_outputs,
            )
