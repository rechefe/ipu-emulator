"""Stride-2 depthwise 3x3 convolution for small spatial sizes (cols <= 64).

Input:  rows x cols x channels (INT8, cols in {32, 64})
Output: (rows/2) x (cols/2) x channels (INT8, quantized)

Output is stored in the same interleaved chunk format as input, so it
can be consumed directly by a subsequent convolution layer.

The assembly is a Jinja template parameterized by ``cols``.  The harness
injects ``{% set cols = XX %}`` before passing the source to the assembler.

Usage::

    from ipu_apps.convolutions_universal.depthwise_conv_stride2_small import (
        DepthwiseConvStride2SmallApp,
    )

    app = DepthwiseConvStride2SmallApp(
        inst_path="stride2_small.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        dtype="INT8",
        rows=64, cols=64, channels=16,
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
KERNEL_BASE_ADDR = 0x110000
MASK_BASE_ADDR = 0x120000
TEMP_BASE_ADDR = 0x120080  # 512 bytes for 4 intermediate conv results (4 x 128)
ZERO_BASE_ADDR = 0x120280  # 128 bytes of zeros for bottom-border S2
OUTPUT_BASE_ADDR = 0x130000

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

    Slots:
      0: all zeros      (no masking, kc=0)
      1: left border    (zero col 0 of each packed row, kc=-1)
      2: right border   (zero last col of each packed row, kc=+1)
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


class DepthwiseConvStride2SmallApp(IpuApp):
    """Stride-2 depthwise 3x3 convolution for cols <= 64.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary.
        kernel_path:  Path to kernel binary.
        output_path:  Optional path to write output.
        dtype:        Data type string or DType (INT8 only for now).
        rows:         Spatial height of input.
        cols:         Spatial width of input (32 or 64).
        channels:     Number of channels (must be multiple of 8).
    """

    # Path to the Jinja assembly template (sibling file)
    ASM_TEMPLATE_PATH = Path(__file__).resolve().parent / "depthwise_conv_stride2_small.asm"

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
        valid_cols = {32, 64}
        if cols not in valid_cols:
            raise ValueError(f"cols must be in {valid_cols}, got {cols}")
        if rows % 2 != 0:
            raise ValueError(f"rows must be even (got {rows})")
        if channels % 8 != 0 or channels < 8:
            raise ValueError(f"channels must be a multiple of 8 and >= 8 (got {channels})")

        num_chunks = (rows * cols) // 128
        num_groups = num_chunks // 4
        if num_groups < 2:
            raise ValueError(
                f"Need at least 2 groups of 4 chunks (rows*cols >= 1024), "
                f"got {rows}*{cols}={rows*cols} ({num_chunks} chunks, {num_groups} groups)"
            )

        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.num_chunks = num_chunks
        self.num_groups = num_groups
        self.out_rows = rows // 2
        self.out_cols = cols // 2
        self.group_stride = channels * 128
        self.kernel_groups = channels // 8

    def get_asm_source(self) -> str:
        """Return the Jinja-rendered assembly source for the configured cols."""
        template_src = self.ASM_TEMPLATE_PATH.read_text()
        return f"{{% set cols = {self.cols} %}}\n" + template_src

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))

        # Load input
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (packed)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = _build_kernel_data(kernel_raw, self.channels)
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_padded)

        # Load mask data
        mask_data = _build_mask_data(self.cols)
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # Write zeros for bottom-border S2 loading
        state.xmem.write_address(ZERO_BASE_ADDR, bytes(128))

        # CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)
        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(5, self.num_chunks)
        state.regfile.set_cr(6, self.group_stride)
        state.regfile.set_cr(7, 1024)  # channel group size = 8 * 128
        state.regfile.set_cr(8, TEMP_BASE_ADDR)
        state.regfile.set_cr(9, 1)  # identity scalar for mult.ve.cr
        state.regfile.set_cr(10, ZERO_BASE_ADDR)
        state.regfile.set_cr(11, self.num_groups - 1)  # last group index
        state.regfile.set_cr(12, 128)  # step constant for add

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_out_bytes = self.num_groups * self.channels * 128
            data = state.xmem.read_address(OUTPUT_BASE_ADDR, total_out_bytes)
            Path(self.output_path).write_bytes(data)
