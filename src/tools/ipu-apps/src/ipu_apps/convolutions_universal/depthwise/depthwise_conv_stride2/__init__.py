"""Stride-2 depthwise 3x3 convolution harness.

Input: rows x cols x channels (INT8, cols=128)
Output: (rows/2) x (cols/2) x channels (INT8, quantized)

The convolution centers at input rows 1, 3, 5, ..., producing half the
spatial dimensions.  Each output chunk packs 2 output rows (2 x 64 = 128
bytes).  The last output chunk only has 1 valid row (bottom border skipped).

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2 import (
        DepthwiseConvStride2App,
    )

    app = DepthwiseConvStride2App(
        inst_path="depthwise_conv_stride2.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        dtype="INT8",
        rows=128, cols=128, channels=16,
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

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x000000


def _compute_addresses(rows: int, cols: int, channels: int) -> dict:
    """Compute memory addresses dynamically to avoid overlap for large configs."""
    def align(addr: int, boundary: int = 256) -> int:
        return (addr + boundary - 1) & ~(boundary - 1)

    input_size = rows * channels * 128
    kernel_base = align(INPUT_BASE_ADDR + input_size)
    kernel_size = (channels // 8) * 128
    mask_base = align(kernel_base + kernel_size)
    temp_base = mask_base + 128  # mask is 128 bytes, temp follows immediately
    output_base = align(temp_base + 256)  # temp is 256 bytes (temp_A + temp_B)
    return {
        "kernel": kernel_base,
        "mask": mask_base,
        "temp": temp_base,
        "output": output_base,
    }

_DTYPE_MAP = {
    "INT8": DType.INT8,
    "int8": DType.INT8,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Currently only INT8 supported.")
    return dt


def _build_mask_data(cols: int) -> bytes:
    """Build 128-byte mask register data for a given spatial width.

    Same mask layout as the non-strided depthwise conv:
      slot 0: all zeros (no masking, kc=0)
      slot 1: left border (zero col 0 of each packed row, kc=-1)
      slot 2: right border (zero last col of each packed row, kc=+1)
    """
    rows_per_chunk = 128 // cols
    mask = bytearray(128)

    def set_bit(slot: int, bit: int) -> None:
        byte_idx = slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    for r in range(rows_per_chunk):
        set_bit(1, r * cols)
    for r in range(rows_per_chunk):
        set_bit(2, r * cols + cols - 1)

    return bytes(mask)


def _build_kernel_data(kernel_raw: bytes, channels: int) -> bytes:
    """Pack kernel into groups of 128 bytes (8 channels per group, padded)."""
    kernel_groups = channels // 8
    kernel_padded = bytearray(kernel_groups * 128)
    for g in range(kernel_groups):
        src_offset = g * 8 * 9
        dst_offset = g * 128
        kernel_padded[dst_offset:dst_offset + 72] = kernel_raw[src_offset:src_offset + 72]
    return bytes(kernel_padded)


class DepthwiseConvStride2App(IpuApp):
    """Stride-2 depthwise 3x3 convolution application harness.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary.
        kernel_path:  Path to kernel binary.
        output_path:  Optional path to write output.
        dtype:        Data type string or DType (INT8 only for now).
        rows:         Spatial height of input (must be even).
        cols:         Spatial width of input (must be 128 for now).
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

        if cols != 128:
            raise ValueError(f"cols must be 128 for stride-2 (got {cols})")
        if rows % 2 != 0:
            raise ValueError(f"rows must be even (got {rows})")
        if channels % 8 != 0 or channels < 8:
            raise ValueError(f"channels must be a multiple of 8 and >= 8 (got {channels})")

        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.num_input_chunks = (rows * cols) // 128  # per channel
        self.num_output_chunks = rows // 4  # output_rows/2 = (rows/2)/2
        self.out_rows = rows // 2
        self.out_cols = cols // 2
        self.group_stride = channels * 128
        self.kernel_groups = channels // 8

        addrs = _compute_addresses(rows, cols, channels)
        self._kernel_base = addrs["kernel"]
        self._mask_base = addrs["mask"]
        self._temp_base = addrs["temp"]
        self.output_base = addrs["output"]

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))

        # Load input
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (packed)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = _build_kernel_data(kernel_raw, self.channels)
        state.xmem.write_address(self._kernel_base, kernel_padded)

        # Load mask data
        mask_data = _build_mask_data(self.cols)
        state.xmem.write_address(self._mask_base, mask_data)

        # CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, self._kernel_base)
        state.regfile.set_cr(2, self.output_base)
        state.regfile.set_cr(3, self._mask_base)
        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(5, self.num_output_chunks)
        state.regfile.set_cr(6, self.group_stride)
        state.regfile.set_cr(7, 1024)  # channel group size = 8 * 128
        state.regfile.set_cr(8, self._temp_base)
        state.regfile.set_cr(9, 1)  # identity scalar for mult.ve.cr
        state.regfile.set_cr(10, 0)   # zero constant
        state.regfile.set_cr(11, 1)   # small constant 1
        state.regfile.set_cr(12, 128)  # step constant for add
        state.regfile.set_cr(13, 2)   # small constant 2
        state.regfile.set_cr(14, 31)  # small constant 31

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_chunks = self.num_output_chunks * self.channels
            total_bytes = total_chunks * 128
            data = state.xmem.read_address(self.output_base, total_bytes)
            Path(self.output_path).write_bytes(data)
