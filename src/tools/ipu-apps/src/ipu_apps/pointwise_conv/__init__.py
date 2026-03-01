"""Pointwise (1x1) convolution harness.

128x128x8 input, 1x1x8x8 kernel, 128x128x8 output.
Exercises pipelined r_cyclic loading (8 input channels > 4-slot capacity).

Usage::

    from ipu_apps.pointwise_conv import PointwiseConvApp

    app = PointwiseConvApp(
        inst_path="pointwise_conv.bin",
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

ROWS = 128
COLS = 128
IN_CHANNELS = 8
OUT_CHANNELS = 8

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x20000
MASK_BASE_ADDR = 0x30000
OUTPUT_BASE_ADDR = 0x40000

# Output: 128 rows x 8 channels x 512 bytes (128 elements x 4-byte accumulator)
OUTPUT_ROW_BYTES = COLS * 4  # 512 per channel-row

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


class PointwiseConvApp(IpuApp):
    """Pointwise (1x1) convolution application harness.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input image binary (128x128x8 bytes, interleaved by row).
        kernel_path: Path to kernel binary (64 bytes, padded to 128).
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

        # Load input image (128 rows x 8 channels x 128 bytes = 131072 bytes)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (padded to 128 bytes, only first 64 matter)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = bytearray(COLS)
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
            # Output: 128 rows x 8 channels, each 512 bytes
            total_rows = ROWS * OUT_CHANNELS
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_ROW_BYTES, total_rows,
            )
