"""Universal depthwise 3x3 convolution harness.

A single parameterized harness that works for ANY valid depthwise convolution
configuration (spatial >= 16x16, channels multiple of 8).

Dimensions are passed at construction time; the harness computes all derived
constants, builds the correct masks for the spatial size, and sets CR4-CR7.

Usage::

    from ipu_apps.convolutions_universal.depthwise_conv_universal import DepthwiseConvUniversalApp

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

OUTPUT_CHUNK_BYTES = 128 * 4  # 512 bytes per output channel per chunk


def _build_kernel_data(kernel_raw: bytes, channels: int) -> bytes:
    """Pack kernel into groups of 128 bytes (8 channels per group, padded).

    Input: channels * 9 bytes (contiguous, 9 taps per channel).
    Output: (channels/8) groups of 128 bytes each.
    """
    kernel_groups = channels // 8
    kernel_padded = bytearray(kernel_groups * 128)
    for g in range(kernel_groups):
        src_offset = g * 8 * 9  # 72 bytes per group
        dst_offset = g * 128
        kernel_padded[dst_offset:dst_offset + 72] = kernel_raw[src_offset:src_offset + 72]
    return bytes(kernel_padded)


class DepthwiseConvUniversalApp(IpuApp):
    """Universal depthwise 3x3 convolution application harness.

    Args:
        inst_path:    Path to assembled universal binary.
        input_path:   Path to input image binary.
        kernel_path:  Path to kernel binary.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height.
        cols:         Spatial width (power of 2, 16-128).
        channels:     Number of channels (must be multiple of 8).
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

        # Validate
        valid_cols = {16, 32, 64, 128}
        if cols not in valid_cols:
            raise ValueError(f"cols must be in {valid_cols}, got {cols}")
        num_chunks = (rows * cols) // 128
        if num_chunks < 2:
            raise ValueError(
                f"Need at least 2 chunks (rows*cols >= 256), got {rows}*{cols}={rows*cols}"
            )
        if channels % 8 != 0:
            raise ValueError(f"channels ({channels}) must be a multiple of 8")
        if channels < 8:
            raise ValueError(f"channels ({channels}) must be >= 8")

        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.num_chunks = num_chunks
        self.group_stride = channels * 128
        self.kernel_groups = channels // 8

    def setup(self, state: "IpuState") -> None:
        # Set data type
        state.set_cr_dtype(int(self.dtype))

        # Load input
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (packed into groups of 128 bytes)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = _build_kernel_data(kernel_raw, self.channels)
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_padded)

        # Load mask data (computed from cols)
        mask_data = build_border_mask_data(self.cols)
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # Write 128 bytes of zeros for S2 zero-loading in last chunk
        state.xmem.write_address(ZERO_BASE_ADDR, bytes(128))

        # Set base-address CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)

        # Set parameter CR registers
        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(5, self.num_chunks)
        state.regfile.set_cr(6, self.group_stride)
        state.regfile.set_cr(7, 1024)  # channel group size = 8 * 128
        state.regfile.set_cr(8, ZERO_BASE_ADDR)  # zero region for S2 in last chunk

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.channels
            dump_outputs(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_outputs,
            )
