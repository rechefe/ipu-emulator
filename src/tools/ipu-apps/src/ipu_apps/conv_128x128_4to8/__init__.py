"""Standard 3x3 convolution harness (4 input -> 8 output channels, 128x128).

Each output channel has 4 input-channel 3x3 kernels (full cross-channel mixing).
Kernels are stored at 128-byte XMEM boundaries (one filter per 128-byte slot).

Usage::

    from ipu_apps.conv_128x128_4to8 import Conv128x128_4to8App

    app = Conv128x128_4to8App(
        inst_path="conv_128x128_4to8.bin",
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
IN_CHANNELS = 4
OUT_CHANNELS = 8
KERNEL_SIZE = 9  # 3x3

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x10000
MASK_BASE_ADDR = 0x20000
OUTPUT_BASE_ADDR = 0x30000

# Each filter: 36 bytes of data, padded to 128 bytes
FILTER_PADDED_BYTES = 128
KERNEL_TOTAL_BYTES = OUT_CHANNELS * FILTER_PADDED_BYTES  # 1024

# Output: 128 elements x 4-byte accumulator per channel-row
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
      slot 0: all zeros       -- no masking (for dc=0)
      slot 1: bit 0 set       -- zero position 0 (for dc=-1, left border)
      slot 2: bit 127 set     -- zero position 127 (for dc=+1, right border)
      slots 3-7: unused (zeros)
    """
    mask = bytearray(128)
    mask[16] = 0x01
    mask[32 + 15] = 0x80
    return bytes(mask)


class Conv128x128_4to8App(IpuApp):
    """Standard 3x3 convolution application harness (4 -> 8 channels).

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input binary (128x128x4 bytes, interleaved by row).
        kernel_path: Path to kernel binary (1024 bytes: 8 filters x 128 bytes each).
        output_path: Optional path to write output.
        dtype:       Data type string or :class:`DType`.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))

        # Load input (128 rows x 4 channels x 128 bytes = 65536 bytes)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (8 filters x 128 bytes = 1024 bytes, already padded)
        kernel_data = self.kernel_path.read_bytes()
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_data)

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
            total_rows = ROWS * OUT_CHANNELS
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_ROW_BYTES, total_rows,
            )
