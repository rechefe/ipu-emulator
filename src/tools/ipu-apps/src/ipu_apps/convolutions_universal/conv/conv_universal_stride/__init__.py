"""Universal standard 3x3 convolution, stride 2, cols=128 (milestone 1).

Same walking-pointer / role-rotating 9-cyc-per-channel body as
``conv_universal``, restricted to ``cols == 128`` (one input spatial row ==
one 128-byte chunk). The outer loop visits only even input rows as
convolution centers (stride 2, pad 1); since ``rows`` must be even, the last
center row's kr=+1 neighbour is always in-bounds, so only the top border
(mask slot 3) is needed -- no bottom-border case.

Output rows are 64 lanes wide (``cols // 2``), so two consecutive output
rows pack into one 128-byte output chunk. Each row's 64-lane decimated
result is staged through a TEMP xmem scratch region before the pair is
combined and stored (see conv_universal_stride.asm's header comment for the
full reload-through-temp mechanics).

Usage::

    from ipu_apps.convolutions_universal.conv.conv_universal_stride import (
        ConvUniversalStrideApp,
    )

    app = ConvUniversalStrideApp(
        inst_path="conv_universal_stride.bin",
        input_path="input.bin",
        kernel=weights_nhwc,      # np.ndarray [out_ch, in_ch, 3, 3]
        output_path="output.bin",
        dtype="INT8",
        rows=8, cols=128, in_channels=16, out_channels=16,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_math import DType
from ipu_emu.ipu_config import Partition

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    CHUNK_BYTES,
    parse_dtype,
    dump_outputs,
)
from ipu_apps.convolutions_universal.conv.conv_universal import (
    FPB,
    SUPER_BLOCK_BYTES,
    MASK_SLOT_TOP,
    _pack_conv_weights_fpb28,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x100000
MASK_BASE_ADDR = 0x180000
TEMP_BASE_ADDR = 0x1A0000     # 128-byte scratch, reused per output-row pair
OUTPUT_BASE_ADDR = 0x1C0000

OUTPUT_CHUNK_BYTES = CHUNK_BYTES  # 128: two 64-lane output rows packed per chunk

COLS = 128  # milestone 1: cols=128 only


def build_top_border_mask_blob() -> bytes:
    """Build the 128-byte R_MASK blob for cols=128 (single packed row/chunk).

    Only slot 3 (top-row zero, for pair 0's row 0 g0-body kr=-1 taps) is
    needed -- unlike conv_universal there is no bottom-border (gN) case,
    since the outer loop only visits even rows and the last center row's
    kr=+1 neighbour is always in-bounds. Slot 0 (all-KEEP) is unused by this
    asm but included for parity/safety.
    """
    mask = bytearray(128)
    all_keep = bytes([0xFF] * 16)
    mask[0:16] = all_keep       # slot 0: none (unused, kept for parity)
    # slot 3: zero every lane (whole packed row is out-of-bounds)
    mask[3 * 16:3 * 16 + 16] = bytes(16)
    return bytes(mask)


class ConvUniversalStrideApp(IpuApp):
    """Universal standard 3x3 convolution, stride 2, cols=128.

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    Args:
        inst_path:    Path to assembled conv_universal_stride binary.
        input_path:   Path to input image binary (chunk-interleaved layout,
                      one 128-wide spatial row per 128-byte chunk).
        kernel:       Numpy weights of shape ``[out_ch, in_ch, 3, 3]``.
        kernel_path:  Alternative: path to a raw ``[out_ch, in_ch, 9]``
                      contiguous byte file.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height of the INPUT (must be even; output has
                      rows // 2 rows).
        cols:         Spatial width; must be 128 (milestone 1).
        in_channels:  Number of input channels (>= 1).
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
        kernel: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

        kernel_path = getattr(self, "kernel_path", None)
        if kernel is not None and kernel_path is not None:
            raise ValueError("Provide exactly one of kernel= or kernel_path=")
        if kernel is None and kernel_path is None:
            raise ValueError("Provide one of kernel= or kernel_path=")
        self._kernel_array = kernel
        self.kernel_path = Path(kernel_path) if kernel_path is not None else None

        if cols != COLS:
            raise ValueError(
                f"conv_universal_stride (milestone 1) only supports cols={COLS}, got {cols}"
            )
        if rows < 2 or rows % 2 != 0:
            raise ValueError(f"rows must be even and >= 2, got {rows}")
        if in_channels < 1:
            raise ValueError(f"in_channels ({in_channels}) must be >= 1")
        if out_channels < 1:
            raise ValueError(f"out_channels ({out_channels}) must be >= 1")

        self.rows = rows
        self.cols = cols
        self.out_rows = rows // 2
        self.out_cols = cols // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        # One pair_loop iteration handles 2 output rows (packed into 1 chunk).
        self.num_pairs = self.out_rows // 2
        self.in_group_stride = in_channels * CHUNK_BYTES
        self.blocks_per_filter = -(-in_channels // FPB)  # ceil
        self.total_kernel_bytes = out_channels * self.blocks_per_filter * SUPER_BLOCK_BYTES

    def _pack_kernel(self) -> bytes:
        if self._kernel_array is not None:
            weights = self._kernel_array
        else:
            raw = self.kernel_path.read_bytes()
            expected = self.out_channels * self.in_channels * 9
            if len(raw) != expected:
                raise ValueError(
                    f"kernel_path file has {len(raw)} bytes, "
                    f"expected {expected} (out_ch * in_ch * 9)"
                )
            if self.dtype != DType.INT8:
                raise ValueError(
                    "kernel_path is only supported for INT8; use kernel= "
                    "for FP8 dtypes"
                )
            weights = (
                np.frombuffer(raw, dtype=np.int8)
                .reshape(self.out_channels, self.in_channels, 3, 3)
            )
        w_reordered = weights.reshape(self.out_channels, self.in_channels, 9)
        return _pack_conv_weights_fpb28(w_reordered, self.dtype)

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype

        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        kernel_packed = self._pack_kernel()
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_packed)

        state.xmem.write_address(MASK_BASE_ADDR, build_top_border_mask_blob())

        # CR15 dstructure: cols=128 -> one packed spatial row per chunk, no
        # sub-row packing (single partition group covering all 128 lanes).
        state.set_cr_dstructure(
            valid_elements=128,
            partition=Partition.P0,
        )

        # CR register map -- see conv_universal_stride.asm's header comment
        # for the authoritative list.
        state.regfile.set_cr(10, INPUT_BASE_ADDR)
        state.regfile.set_cr(5, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)
        state.regfile.set_cr(9, TEMP_BASE_ADDR)

        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(6, self.in_group_stride)
        state.regfile.set_cr(7, FPB * CHUNK_BYTES)
        state.regfile.set_cr(8, self.total_kernel_bytes)

        # cr11 = pair-loop limit = (num_pairs - 1) * (2 * in_group_stride).
        # lr_chunk_base advances by cr_group_stride (=in_group_stride) once
        # per ROW (row0->row1 in row0_drain, row1->next pair's row0 at the
        # bottom of the pair loop), i.e. by 2*in_group_stride per pair.
        state.regfile.set_cr(
            11, (self.num_pairs - 1) * (2 * self.in_group_stride)
        )

        state.regfile.set_cr(12, CHUNK_BYTES)
        state.regfile.set_cr(13, SUPER_BLOCK_BYTES)
        # cr14 = end-of-9 walking-pointer wrap step = ring_adv(384) - 2*cols - 2
        state.regfile.set_cr(14, (3 * CHUNK_BYTES - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_pairs * self.out_channels
            dump_outputs(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_outputs,
            )
