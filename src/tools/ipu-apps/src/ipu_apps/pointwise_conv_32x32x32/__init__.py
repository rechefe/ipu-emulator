"""Pointwise (1x1) convolution harness — 32x32x32 variant.

32x32x32 input, 1x1x32x32 kernel, 32x32x32 output.
Since spatial width (32) < SIMD width (128), 4 spatial rows are packed
per 128-byte chunk.  32 input/output channels with 1024-byte kernel split
across 4 kernel groups of r0+r1 (256 bytes each).

Usage::

    from ipu_apps.pointwise_conv_32x32x32 import PointwiseConv32x32x32App

    app = PointwiseConv32x32x32App(
        inst_path="pointwise_conv_32x32x32.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        dtype="INT8",
    )
    state, cycles = app.run()
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
IN_CHANNELS = 32
OUT_CHANNELS = 32

# 4 spatial rows pack into one 128-byte chunk (128 / 32 = 4)
ROWS_PER_CHUNK = 128 // COLS  # 4
ROW_GROUPS = ROWS // ROWS_PER_CHUNK  # 8

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x20000
MASK_BASE_ADDR = 0x30000
OUTPUT_BASE_ADDR = 0x40000

KERNEL_BYTES = IN_CHANNELS * OUT_CHANNELS  # 1024

# Output: each store writes 128 elements x 4 bytes = 512 bytes (full SIMD width)
OUTPUT_ROW_BYTES = 128 * 4  # 512 per output-channel per row-group

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
    """Build the 128-byte mask register data (all zeros — no masking)."""
    return bytes(128)


class PointwiseConv32x32x32App(IpuApp):
    """Pointwise (1x1) convolution application harness — 32x32x32 variant.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input image binary (32x32x32 bytes, interleaved by row-group).
        kernel_path: Path to kernel binary (1024 bytes).
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

        # Load input image (8 row-groups x 32 channels x 128 bytes = 32768 bytes)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (1024 bytes)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = bytearray(KERNEL_BYTES)
        kernel_padded[:len(kernel_raw)] = kernel_raw
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_padded)

        # Load mask data (all zeros)
        mask_data = _build_mask_data()
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # Set CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, MASK_BASE_ADDR)
        state.regfile.set_cr(3, OUTPUT_BASE_ADDR)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # Output: 8 row-groups x 32 channels, each 512 bytes
            total_rows = ROW_GROUPS * OUT_CHANNELS
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_ROW_BYTES, total_rows,
            )
