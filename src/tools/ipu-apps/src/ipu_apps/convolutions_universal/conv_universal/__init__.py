"""Universal standard 3x3 convolution harness.

A single parameterized harness that works for ANY valid standard convolution
configuration (spatial >= 16x16, in_channels multiple of 8, >= 8).

Dimensions are passed at construction time; the harness computes all derived
constants, builds the correct masks for the spatial size, packs kernels into
128-byte blocks, and sets CR0-CR8.

Usage::

    from ipu_apps.convolutions_universal.conv_universal import ConvUniversalApp

    app = ConvUniversalApp(
        inst_path="conv_universal.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        dtype="INT8",
        rows=32, cols=32, in_channels=16, out_channels=16,
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

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x200000
MASK_BASE_ADDR = 0x600000
OUTPUT_BASE_ADDR = 0x700000

OUTPUT_CHUNK_BYTES = 128 * 4  # 512 bytes per output channel per chunk

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


def _build_mask_data(cols: int) -> bytes:
    """Build the 128-byte mask register data for a given spatial width.

    Layout (8 slots of 16 bytes = 128 bits each):
      slot 0: all zeros         -> no masking (kc=0)
      slot 1: left border       -> zero col 0 of each packed row (kc=-1)
      slot 2: right border      -> zero last col of each packed row (kc=+1)
      slot 3: bottom row only   -> zero last spatial row in chunk
                                   (last chunk, kr=+1, kc=0)
      slot 4: left + bottom     -> union of slot 1 and slot 3
                                   (last chunk, kr=+1, kc=-1)
      slot 5: right + bottom    -> union of slot 2 and slot 3
                                   (last chunk, kr=+1, kc=+1)
      slots 6-7: unused (zeros)
    """
    rows_per_chunk = 128 // cols
    mask = bytearray(128)

    def set_bit(slot: int, bit: int) -> None:
        byte_idx = slot * 16 + bit // 8
        mask[byte_idx] |= 1 << (bit % 8)

    def set_bit_range(slot: int, lo: int, hi: int) -> None:
        for b in range(lo, hi + 1):
            set_bit(slot, b)

    # Slot 1: left border — zero col 0 of each packed row
    for r in range(rows_per_chunk):
        set_bit(1, r * cols)

    # Slot 2: right border — zero last col of each packed row
    for r in range(rows_per_chunk):
        set_bit(2, r * cols + cols - 1)

    # Slot 3: bottom row — zero last spatial row positions in chunk
    last_row_start = (rows_per_chunk - 1) * cols
    set_bit_range(3, last_row_start, 127)

    # Slot 4: left border + bottom row
    for r in range(rows_per_chunk):
        set_bit(4, r * cols)
    set_bit_range(4, last_row_start, 127)

    # Slot 5: right border + bottom row
    for r in range(rows_per_chunk):
        set_bit(5, r * cols + cols - 1)
    set_bit_range(5, last_row_start, 127)

    return bytes(mask)


def _build_kernel_data(
    kernel_raw: bytes,
    in_channels: int,
    out_channels: int,
) -> bytes:
    """Pack kernel into blocks of 128 bytes (8 input channels per block).

    Input: out_channels * in_channels * 9 bytes (contiguous).
      kernel_raw[f * in_channels * 9 + ic * 9 + tap]

    Output: out_channels * (in_channels // 8) blocks of 128 bytes each.
      Each block: 8 input channels x 9 taps = 72 bytes + 56 padding.
    """
    blocks_per_filter = in_channels // 8
    taps_per_block = 8 * 9  # 72 bytes
    taps_per_filter = in_channels * 9
    total_blocks = out_channels * blocks_per_filter
    kernel_padded = bytearray(total_blocks * 128)

    for f in range(out_channels):
        for b in range(blocks_per_filter):
            src_offset = f * taps_per_filter + b * taps_per_block
            dst_offset = (f * blocks_per_filter + b) * 128
            kernel_padded[dst_offset:dst_offset + taps_per_block] = (
                kernel_raw[src_offset:src_offset + taps_per_block]
            )

    return bytes(kernel_padded)


class ConvUniversalApp(IpuApp):
    """Universal standard 3x3 convolution application harness.

    Args:
        inst_path:    Path to assembled universal binary.
        input_path:   Path to input image binary.
        kernel_path:  Path to kernel binary.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height.
        cols:         Spatial width (power of 2, 16-128).
        in_channels:  Number of input channels (must be multiple of 8).
        out_channels: Number of output channels (>= 1).
    """

    def __init__(
        self,
        *,
        dtype: str | DType = "INT8",
        rows: int,
        cols: int,
        in_channels: int,
        out_channels: int,
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
        if in_channels % 8 != 0:
            raise ValueError(f"in_channels ({in_channels}) must be a multiple of 8")
        if in_channels < 8:
            raise ValueError(f"in_channels ({in_channels}) must be >= 8")
        if out_channels < 1:
            raise ValueError(f"out_channels ({out_channels}) must be >= 1")

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_chunks = num_chunks
        self.in_group_stride = in_channels * 128
        self.blocks_per_filter = in_channels // 8
        self.total_kernel_bytes = out_channels * self.blocks_per_filter * 128

    def setup(self, state: "IpuState") -> None:
        # Set data type
        state.set_cr_dtype(int(self.dtype))

        # Load input
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Load kernel (packed into blocks of 128 bytes)
        kernel_raw = self.kernel_path.read_bytes()
        kernel_padded = _build_kernel_data(
            kernel_raw, self.in_channels, self.out_channels,
        )
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_padded)

        # Load mask data (computed from cols)
        mask_data = _build_mask_data(self.cols)
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # Set base-address CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)

        # Set parameter CR registers
        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(5, self.num_chunks)
        state.regfile.set_cr(6, self.in_group_stride)
        state.regfile.set_cr(7, 1024)  # channel group size = 8 * 128
        state.regfile.set_cr(8, self.total_kernel_bytes)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.out_channels
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_outputs,
            )
