"""Universal standard 3x3 convolution harness.

A single parameterized harness that works for ANY valid standard convolution
configuration (spatial >= 16x16, in_channels >= 1).

Dimensions are passed at construction time; the harness computes all derived
constants, builds the correct masks for the spatial size, packs kernels into
dense FPB=14 blocks, and sets CR0-CR9.

Kernel super-block layout (FPB=28):
  One 256-byte super-block holds up to 28 input-channel slots of one output
  filter, packed as two 128-byte halves: channels 0..13 in the first half
  (loaded into R0) and channels 14..27 in the second half (loaded into R1).
  Each slot is 9 bytes (3x3 taps, row-major): 28 * 9 = 252 real bytes + 4 pad.
  The asm uses the shared-index `mult.ve` (fixed_idx 0..255) to address all
  28 channels with a single index. Per filter: ceil(in_channels / 28) super-
  blocks, contiguous. The last super-block of each filter zero-pads any
  unused slots; the asm clamps the inner loop to the real channel count on
  that block (min(lr10+cr7, in_group_stride)).

Usage::

    from ipu_apps.convolutions_universal.conv.conv_universal import ConvUniversalApp

    # Numpy weights form (preferred):
    app = ConvUniversalApp(
        inst_path="conv_universal.bin",
        input_path="input.bin",
        kernel=weights_nhwc,      # np.ndarray [out_ch, in_ch, 3, 3]
        output_path="output.bin",
        dtype="INT8",
        rows=32, cols=32, in_channels=16, out_channels=16,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_math import DType

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    parse_dtype,
    build_border_mask_data,
    dump_outputs,
)
from ipu_apps.convolutions_universal.weights import (  # noqa: F401
    pack_conv_weights_dense,  # kept for API compat
    _validate_and_cast_to_bytes,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x200000
MASK_BASE_ADDR = 0x600000
ZERO_BASE_ADDR = 0x600080  # 128 bytes of zeros (right after mask data)
OUTPUT_BASE_ADDR = 0x700000

OUTPUT_CHUNK_BYTES = 128  # 128 bytes per output filter per chunk (int8)

FPB = 28  # channels per 256-byte super-block (K=3 dense, R0+R1 shared index)
SUPER_BLOCK_BYTES = 256  # = 2 * 128 (R0 half + R1 half)
HALF_FPB = 14            # channels per 128-byte half (R0 or R1)


def _pack_conv_weights_fpb28(
    weights_reordered: np.ndarray, dtype: DType
) -> bytes:
    """Pack [out_ch, in_ch, 9] (taps already reordered) into FPB=28 super-blocks.

    Each super-block is 256 bytes laid out *linearly*: channel `s` occupies
    bytes [s*9 .. s*9+9). The first 128 bytes are loaded into R0 and the
    second 128 bytes into R1; the asm uses mult.ve with a shared fixed_idx
    (0..255) that sweeps the entire super-block, so a channel whose weights
    cross the 128-byte boundary (e.g. channel 14 starts at byte 126) is
    served partly from R0 and partly from R1 transparently.

    Per-filter byte stride: ceil(in_ch / 28) * 256.
    """
    out_ch, in_ch, k2 = weights_reordered.shape
    if k2 != 9:
        raise ValueError(f"expected last dim=9 (taps), got {k2}")
    raw = _validate_and_cast_to_bytes(
        weights_reordered.reshape(out_ch, in_ch, 3, 3), dtype
    )
    # raw indexing: byte for filter f, channel ic, tap t = raw[(f*in_ch+ic)*9 + t]

    super_blocks_per_filter = math.ceil(in_ch / FPB)
    total = out_ch * super_blocks_per_filter * SUPER_BLOCK_BYTES
    packed = bytearray(total)

    for f in range(out_ch):
        for sb in range(super_blocks_per_filter):
            sb_base = (f * super_blocks_per_filter + sb) * SUPER_BLOCK_BYTES
            for s in range(FPB):
                ic = sb * FPB + s
                if ic >= in_ch:
                    break
                src = (f * in_ch + ic) * 9
                dst = sb_base + s * 9   # linear layout: 0,9,18,...,243
                packed[dst:dst + 9] = raw[src:src + 9]
    return bytes(packed)


class ConvUniversalApp(IpuApp):
    """Universal standard 3x3 convolution application harness.

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    Args:
        inst_path:    Path to assembled universal binary.
        input_path:   Path to input image binary (chunk-interleaved layout).
        kernel:       Numpy weights of shape ``[out_ch, in_ch, 3, 3]``.
                      Packed at setup via :func:`pack_conv_weights_dense`.
        kernel_path:  Alternative: path to a raw ``[out_ch, in_ch, 9]``
                      contiguous byte file. Reshaped to ``[out_ch, in_ch, 3, 3]``
                      and packed the same way.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height.
        cols:         Spatial width (power of 2, 16-128).
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

        # Validate
        # cols=128 is not yet supported by the walking-pointer asm; this
        # binary handles 16/32/64. (cols=128 will live in a separate binary.)
        valid_cols = {16, 32, 64}
        if cols not in valid_cols:
            raise ValueError(f"cols must be in {valid_cols}, got {cols}")
        num_chunks = (rows * cols) // 128
        if num_chunks < 2:
            raise ValueError(
                f"Need at least 2 chunks (rows*cols >= 256), got {rows}*{cols}={rows*cols}"
            )
        if in_channels < 1:
            raise ValueError(f"in_channels ({in_channels}) must be >= 1")
        if out_channels < 1:
            raise ValueError(f"out_channels ({out_channels}) must be >= 1")

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_chunks = num_chunks
        self.in_group_stride = in_channels * 128
        self.blocks_per_filter = math.ceil(in_channels / FPB)
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
            # Raw bytes are assumed to be int8 for INT8 dtype
            # (FP8 rawbytes must be supplied as a kernel= numpy float32 array).
            if self.dtype != DType.INT8:
                raise ValueError(
                    "kernel_path is only supported for INT8; use kernel= "
                    "for FP8 dtypes"
                )
            weights = (
                np.frombuffer(raw, dtype=np.int8)
                .reshape(self.out_channels, self.in_channels, 3, 3)
            )
        # Tap order in the walking-pointer asm: kr=-1 → kr=0 → kr=+1, with
        # kc=-1 → 0 → +1 within each row.  That's natural row-major from the
        # source [out_ch, in_ch, 3, 3] — no reordering needed.
        w_reordered = weights.reshape(self.out_channels, self.in_channels, 9)
        return _pack_conv_weights_fpb28(w_reordered, self.dtype)

    def setup(self, state: "IpuState") -> None:
        # Set data type
        state.set_cr_dtype(int(self.dtype))

        # Load input
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Pack and load kernel (dense FPB=14 layout)
        kernel_packed = self._pack_kernel()
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_packed)

        # Load mask data (computed from cols)
        mask_data = build_border_mask_data(self.cols)
        state.xmem.write_address(MASK_BASE_ADDR, mask_data)

        # Write 128 bytes of zeros for S2 zero-loading in last chunk
        state.xmem.write_address(ZERO_BASE_ADDR, bytes(128))

        # Set base-address CR registers
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)

        # Set parameter CR registers
        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(5, self.num_chunks)
        state.regfile.set_cr(6, self.in_group_stride)
        state.regfile.set_cr(7, FPB * 128)  # channel group size = 28 * 128 = 3584
        state.regfile.set_cr(8, self.total_kernel_bytes)
        state.regfile.set_cr(9, ZERO_BASE_ADDR)  # zero region for S2 in last chunk
        # cr11 = chunk-loop limit = (num_chunks - 1) * in_group_stride
        # Used by asm to compare lr8 (chunk base addr) against chunk limit,
        # replacing the previous lr9 chunk counter.
        state.regfile.set_cr(11, (self.num_chunks - 1) * self.in_group_stride)

        # Constants used by the asm
        state.regfile.set_cr(10, 0)   # zero constant for SET lr<n> cr10
        state.regfile.set_cr(12, 128)
        state.regfile.set_cr(13, 256)
        # cr14 = end-of-9 walking-pointer step: brings lr_walk from this ch's
        # tap-9 offset (lr_read + cols + 1) to next ch's tap-1 offset
        # ((lr_read + 256) - cols - 1), i.e. +(256 - 2*cols - 2).
        state.regfile.set_cr(14, (256 - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.out_channels
            dump_outputs(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_outputs,
            )
