"""Depthwise convolution test harness (1 channel, 128x128, 3x3 kernel).

Mirrors the fully-connected harness pattern:
1. Loads input image, kernel weights, and mask data into XMEM
2. Sets CR registers for base addresses and dtype
3. Runs the assembly program
4. Dumps output from XMEM

Usage::

    from ipu_apps.depthwise_conv import DepthwiseConvApp

    app = DepthwiseConvApp(
        inst_path="depthwise_conv.bin",
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
from ipu_emu.emulator import load_binary_to_xmem, dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants ---------------------------------------------------------------

ROWS = 128
COLS = 128  # = R_REG_SIZE_BYTES
KERNEL_SIZE = 9  # 3x3

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x20000
MASK_BASE_ADDR = 0x30000
OUTPUT_BASE_ADDR = 0x40000

# Output: 128 rows x 512 bytes (128 elements x 4-byte accumulator words)
OUTPUT_ROW_BYTES = COLS * 4  # 512

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
    """Build the 128-byte mask register data.

    Layout (8 slots of 16 bytes = 128 bits each):
      slot 0: all zeros       — no masking (for dc=0)
      slot 1: bit 0 set       — zero position 0 (for dc=-1, left border)
      slot 2: bit 127 set     — zero position 127 (for dc=+1, right border)
      slots 3-7: unused (zeros)
    """
    mask = bytearray(128)

    # Slot 1 (bytes 16..31): bit 0 set
    mask[16] = 0x01

    # Slot 2 (bytes 32..47): bit 127 set
    # Bit 127 within a 128-bit little-endian field = byte 15, bit 7
    mask[32 + 15] = 0x80

    return bytes(mask)


class DepthwiseConvApp(IpuApp):
    """Depthwise convolution application harness.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input image binary (128x128 bytes).
        kernel_path: Path to kernel binary (9 bytes, padded to 128).
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

        # Load input image (128 rows x 128 bytes)
        load_binary_to_xmem(
            state, self.input_path, INPUT_BASE_ADDR, COLS, ROWS
        )

        # Load kernel (padded to 128 bytes, only first 9 matter)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = bytearray(COLS)
        kernel_padded[:len(kernel_raw)] = kernel_raw
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_padded)

        # Load mask data
        mask_data = _build_mask_data()
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # Set CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_ROW_BYTES, ROWS,
            )
