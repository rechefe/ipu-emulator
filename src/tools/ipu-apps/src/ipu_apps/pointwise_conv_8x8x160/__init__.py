"""Pointwise (1x1) convolution harness (160 input -> 160 output channels, 8x8).

Paired-output processing: two output channels share one accumulator.
OC f0 (even) accumulates in lanes 0-63, OC f1 (odd) in lanes 64-127.
One str_acc_reg per pair stores all 512 bytes (both valid).

8x8 spatial with paired IC input layout (same as standard conv):
2 input channels per 128-byte chunk (IC_even in bytes 0-63, IC_odd in 64-127).

Mask: 2 slots packed into one 128-byte register (loaded once):
  Slot 0: bits {64-127} set -> zero lanes 64-127 (for f0)
  Slot 1: bits {0-63} set  -> zero lanes 0-63   (for f1)

Kernel: interleaved per IC pair, 3 blocks of 128 bytes per OC pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants ---------------------------------------------------------------

ROWS = 8
COLS = 8
IN_CHANNELS = 160
OUT_CHANNELS = 160
NUM_PAIRS = OUT_CHANNELS // 2  # 80

INPUT_BASE_ADDR = 0x00000
KERNEL_BASE_ADDR = 0x02800
MASK_BASE_ADDR = 0x0A000
OUTPUT_BASE_ADDR = 0x0A100

# Input: 80 packed chunks x 128 bytes = 10240 bytes
NUM_INPUT_CHUNKS = IN_CHANNELS // 2  # 80

# Kernel: 3 blocks of 128 bytes per OC pair
BLOCKS_PER_PAIR = 3
KERNEL_BLOCK_BYTES = 128
KERNEL_BYTES_PER_PAIR = BLOCKS_PER_PAIR * KERNEL_BLOCK_BYTES  # 384
KERNEL_TOTAL_BYTES = NUM_PAIRS * KERNEL_BYTES_PER_PAIR  # 30720

# Output: 128 elements x 4 bytes = 512 bytes per pair
ACC_CHUNK_BYTES = 512

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
    """Build 128 bytes of mask data: 2 slots in one mask register.

    Slot 0 (bytes 0-15):  bits {64-127} set -> zero lanes 64-127 (for f0)
    Slot 1 (bytes 16-31): bits {0-63} set  -> zero lanes 0-63   (for f1)
    Slots 2-7: zeros (unused).
    """
    mask = bytearray(128)

    def set_bit(slot: int, bit: int) -> None:
        byte_idx = slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    # Slot 0: zero lanes 64-127 (bits 64-127 set)
    for b in range(64, 128):
        set_bit(0, b)

    # Slot 1: zero lanes 0-63 (bits 0-63 set)
    for b in range(0, 64):
        set_bit(1, b)

    return bytes(mask)


class PointwiseConv8x8x160App(IpuApp):
    """Pointwise (1x1) convolution application harness (160->160 channels, 8x8).

    Paired-output processing: OCs are processed in pairs, with f0 (even)
    in lanes 0-63 and f1 (odd) in lanes 64-127 of the same accumulator.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to input binary (10240 bytes: 80 chunks x 128 bytes).
        kernel_path: Path to kernel binary (30720 bytes).
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

        # Load input (80 chunks x 128 bytes = 10240 bytes)
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (30720 bytes)
        kernel_data = self.kernel_path.read_bytes()
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_data)

        # Load mask data (128 bytes)
        mask_data = _build_mask_data()
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # Set CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            result = bytearray()
            for pair in range(NUM_PAIRS):
                addr = OUTPUT_BASE_ADDR + pair * ACC_CHUNK_BYTES
                chunk = state.xmem.read_address(addr, ACC_CHUNK_BYTES)
                result.extend(chunk)
            Path(self.output_path).write_bytes(bytes(result))
