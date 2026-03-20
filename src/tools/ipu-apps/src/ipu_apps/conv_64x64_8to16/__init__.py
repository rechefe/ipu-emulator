"""Standard 3x3 convolution harness (8 input -> 16 output channels, 64x64).

64x64 spatial, 2 rows packed per 128-byte chunk (32 groups).
Each output channel has 8 input-channel 3x3 kernels (72 bytes, fits in r0).
Kernels stored at 128-byte XMEM boundaries (one filter per 128-byte slot).
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

ROWS = 64
COLS = 64
IN_CHANNELS = 8
OUT_CHANNELS = 16

ROWS_PER_CHUNK = 128 // COLS  # 2
NUM_GROUPS = ROWS // ROWS_PER_CHUNK  # 32

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x08000
MASK_BASE_ADDR = 0x09000
OUTPUT_BASE_ADDR = 0x10000

FILTER_PADDED_BYTES = 128
KERNEL_TOTAL_BYTES = OUT_CHANNELS * FILTER_PADDED_BYTES  # 2048

OUTPUT_CHUNK_BYTES = 128 * 4  # 512 (128 elements x 4-byte acc)

_DTYPE_MAP = {
    "INT8": DType.INT8,
    "int8": DType.INT8,
    "FP8_E4M3": DType.FP8_E4M3,
    "fp8_e4m3": DType.FP8_E4M3,
    "FP8_E5M2": DType.FP8_E5M2,
    "fp8_e5m2": DType.FP8_E5M2,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(
            f"Invalid dtype '{dtype_str}'. Supported: INT8, FP8_E4M3, FP8_E5M2"
        )
    return dt


def _build_mask_data() -> bytes:
    """Build the 128-byte mask register data.

    Layout (8 slots of 16 bytes = 128 bits each):
      slot 0: all zeros             -- no masking (kc=0)
      slot 1: bits {0, 64} set      -- zero col 0 of each packed row (kc=-1)
      slot 2: bits {63, 127} set    -- zero col 63 of each packed row (kc=+1)
      slot 3: bits {64..127}        -- zero bottom row (last group kr=+1, kc=0)
      slot 4: bits {0, 64..127}     -- left border + bottom row (last group kr=+1, kc=-1)
      slot 5: bits {63..127}        -- right border + bottom row (last group kr=+1, kc=+1)
      slots 6-7: unused (zeros)
    """
    mask = bytearray(128)

    def set_bit(slot: int, bit: int) -> None:
        byte_idx = slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    def set_bit_range(slot: int, lo: int, hi: int) -> None:
        for b in range(lo, hi + 1):
            set_bit(slot, b)

    # Slot 1: bits 0, 64
    set_bit(1, 0)
    set_bit(1, 64)

    # Slot 2: bits 63, 127
    set_bit(2, 63)
    set_bit(2, 127)

    # Slot 3: bits 64..127
    set_bit_range(3, 64, 127)

    # Slot 4: bits {0} + {64..127}
    set_bit(4, 0)
    set_bit_range(4, 64, 127)

    # Slot 5: bits {63..127}
    set_bit_range(5, 63, 127)

    return bytes(mask)


class Conv64x64_8to16App(IpuApp):
    """Standard 3x3 convolution application harness (8->16 channels, 64x64).

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input binary (32 groups x 8 ch x 128 bytes = 32768 bytes).
        kernel_path: Path to kernel binary (2048 bytes: 16 filters x 128 bytes each).
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

        # Load input (32 groups x 8 channels x 128 bytes = 32768 bytes)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (16 filters x 128 bytes = 2048 bytes, already padded)
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
            total_chunks = NUM_GROUPS * OUT_CHANNELS
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_chunks,
            )
