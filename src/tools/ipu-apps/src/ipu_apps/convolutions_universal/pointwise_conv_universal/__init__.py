"""Universal pointwise (1x1) convolution harness.

A single parameterized harness that works for ANY valid pointwise convolution
configuration. Dimensions are passed at construction time; the harness computes
all derived constants and sets CR registers so the universal assembly binary can
handle any configuration without recompilation.

Supported configurations:
  - Spatial: any power-of-2 rows/cols in [16..128]
  - in_channels: must be divisible by at least 4
  - out_channels: must be divisible by 2 * (128 / G),
    where G = min(largest_pow2_dividing(in_channels), 128)

When in_channels divides 128 (e.g. 4, 8, 16, 32, 64, 128), num_groups == 1
and the assembly uses its original fast path with zero overhead.

When in_channels does NOT divide 128 (e.g. 96, 160, 256, 384, 512), channels
are processed in groups of G, with the accumulator preserved across groups.

Usage::

    from ipu_apps.convolutions_universal.pointwise_conv_universal import PointwiseConvUniversalApp

    app = PointwiseConvUniversalApp(
        inst_path="pointwise_conv_universal.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",
        output_path="output.bin",
        dtype="INT8",
        rows=32, cols=32, in_channels=96, out_channels=384,
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
KERNEL_BASE_ADDR = 0x110000
MASK_BASE_ADDR = 0x120000
OUTPUT_BASE_ADDR = 0x130000

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


def _compute_G(in_channels: int) -> int:
    """Largest power-of-2 dividing in_channels, capped at 128.

    This determines how many input channels are processed per group.
    When G == in_channels (and in_channels divides 128), num_groups == 1
    and the assembly uses its original fast path.
    """
    g = in_channels & (-in_channels)  # isolate lowest set bit
    return min(g, 128)


class PointwiseConvUniversalApp(IpuApp):
    """Universal pointwise (1x1) convolution application harness.

    All dimension parameters are passed at construction time. The harness
    computes derived constants and sets CR registers for the universal assembly.

    Args:
        inst_path:    Path to assembled universal binary.
        input_path:   Path to input image binary.
        kernel_path:  Path to kernel binary.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height (power of 2, 16-128).
        cols:         Spatial width (power of 2, 16-128).
        in_channels:  Number of input channels (must be divisible by >= 4).
        out_channels: Number of output channels.
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

        # Compute channel group size
        G = _compute_G(in_channels)
        if G < 4:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by at least 4"
            )

        num_groups = in_channels // G
        oc_per_reg = 128 // G

        # Validate spatial dimensions
        valid_spatial = {16, 32, 64, 128}
        if rows not in valid_spatial:
            raise ValueError(f"rows must be a power of 2 in {valid_spatial}, got {rows}")
        if cols not in valid_spatial:
            raise ValueError(f"cols must be a power of 2 in {valid_spatial}, got {cols}")

        # Validate out_channels
        if out_channels % (2 * oc_per_reg) != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"{2 * oc_per_reg} (2 * 128/{G})"
            )

        # Store dimensions
        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Channel grouping
        self.G = G
        self.num_groups = num_groups
        self.oc_per_reg = oc_per_reg

        # Derived constants
        self.rows_per_chunk = 128 // cols
        self.row_groups = (rows * cols) // 128
        self.row_group_stride = in_channels * 128
        self.pipeline_limit = G - 5

    def _pack_kernel(self, raw_kernel: bytes) -> bytes:
        """Pack kernel weights for the universal pointwise assembly.

        Raw kernel layout: raw_kernel[oc * in_channels + ic]

        Packed layout (per OC batch, processing 2 * oc_per_reg OCs):
          [r0_group0(128B)][r0_group1(128B)]...[r0_groupN(128B)]
          [r1_group0(128B)][r1_group1(128B)]...[r1_groupN(128B)]

        Within each 128-byte block: oc_per_reg OCs × G bytes.
        OC j, input channel i within the group → byte at offset j * G + i.

        For num_groups == 1 (in_channels divides 128), this produces the same
        layout as the original simple padding.
        """
        G = self.G
        num_groups = self.num_groups
        oc_per_reg = self.oc_per_reg
        in_channels = self.in_channels
        out_channels = self.out_channels
        num_batches = out_channels // (2 * oc_per_reg)

        total_size = num_batches * 2 * num_groups * 128
        packed = bytearray(total_size)

        for batch in range(num_batches):
            for half in range(2):
                for group in range(num_groups):
                    dest_base = (
                        (batch * 2 + half) * num_groups + group
                    ) * 128
                    for j in range(oc_per_reg):
                        oc = batch * 2 * oc_per_reg + half * oc_per_reg + j
                        for i in range(G):
                            ic = group * G + i
                            packed[dest_base + j * G + i] = raw_kernel[
                                oc * in_channels + ic
                            ]

        return bytes(packed)

    def setup(self, state: "IpuState") -> None:
        # Set data type
        state.set_cr_dtype(int(self.dtype))

        # Load input image
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Pack and load kernel
        kernel_raw = self.kernel_path.read_bytes()
        kernel_packed = self._pack_kernel(kernel_raw)
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_packed)

        # Load mask data (all zeros — no masking for pointwise)
        state.xmem.write_address(MASK_BASE_ADDR, bytes(128))

        # Set base-address CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, MASK_BASE_ADDR)
        state.regfile.set_cr(3, OUTPUT_BASE_ADDR)

        # Set parameter CR registers
        state.regfile.set_cr(4, self.oc_per_reg)
        state.regfile.set_cr(5, self.row_groups)
        # pipeline_limit can be negative (-1 for G=4);
        # store as unsigned 32-bit (two's complement)
        state.regfile.set_cr(6, self.pipeline_limit & 0xFFFFFFFF)
        state.regfile.set_cr(7, self.out_channels)
        state.regfile.set_cr(8, self.row_group_stride)

        # Grouped-path CR registers
        state.regfile.set_cr(9, self.num_groups * 128)
        state.regfile.set_cr(10, self.G * 128)
        state.regfile.set_cr(11, self.G)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_rows = self.row_groups * self.out_channels
            dump_xmem_to_binary(
                state,
                self.output_path,
                OUTPUT_BASE_ADDR,
                OUTPUT_ROW_BYTES,
                total_rows,
            )
