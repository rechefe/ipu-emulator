"""Universal standard 3x3 convolution + folded-bias + ReLU harness.

Derived from ``conv_universal``. Same chunk-interleaved I/O layout and FPB=28
super-block kernel packing, with two additions:

  * **Folded bias** — one INT8 bias per output filter, injected as a single
    extra "multiply by 1" accumulate at the start of each filter (see below).
    Batch-norm is assumed already folded into the conv weights + this bias, so
    no separate BN step is needed.
  * **ReLU activation** — applied via ``ACTIVATE relu`` before quantization.

Pipeline per filter: ``r_acc = bias`` (seed), then += 3x3 conv over all input
channels, then ``ACTIVATE relu`` -> ``AAQ`` (INT8 clamp) -> store 128 B.

Kernel super-block layout (FPB=28, +1 bias byte):
  One 256-byte super-block holds up to 28 input-channel slots of one output
  filter. **Byte 0 of every super-block is reserved for the filter's bias**;
  channel ``s`` occupies bytes ``[1 + s*9 .. 1 + s*9 + 9)``. 1 + 28*9 = 253
  bytes <= 256, so capacity is unchanged. The bias byte is written into every
  super-block of a filter for a uniform +1 weight offset, but the asm reads it
  (and accumulates the bias) only once per filter, from super-block 0.
  Channels 0..13 land in the first 128-byte half (R0), 14..27 in the second
  (R1); the shared-index ``mult.ve`` (fixed_idx 0..255) addresses all 28.

Usage::

    from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
        ConvUniversalBnActivationApp,
    )

    # Numpy weights form (preferred):
    app = ConvUniversalBnActivationApp(
        inst_path="conv_universal_bn_activation.bin",
        input_path="input.bin",
        kernel=weights_nhwc,      # np.ndarray [out_ch, in_ch, 3, 3]
        bias=bias_int8,           # np.ndarray [out_ch], folded BN bias (INT8)
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
from ipu_emu.ipu_config import Partition

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    CHUNK_BYTES,
    parse_dtype,
    dump_outputs,
)
from ipu_apps.convolutions_universal.weights import cast_to_wire_bytes

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x100000
# Border handling is done entirely with masks (no zero-region).  A SINGLE
# 128-byte R_MASK blob carries all 3 slots (0=none, 3=top-row zero, 6=bottom-row
# zero); it is loaded once at init.  The vertical out-of-bounds row is zeroed by
# a mask slot instead of loading a zero chunk into the cyclic register; left/
# right edge columns are applied at runtime by mask_shift.
MASK_BASE_ADDR = 0x180000     # single mask blob (slots 0/3/6)
OUTPUT_BASE_ADDR = 0x1C0000

OUTPUT_CHUNK_BYTES = CHUNK_BYTES  # bytes per output filter per chunk (int8)

SUPER_BLOCK_BYTES = 2 * CHUNK_BYTES  # 256 = R0 half + R1 half
HALF_FPB = CHUNK_BYTES // 9          # 14: channels per 128-byte half (9 taps each)
FPB = 2 * HALF_FPB                   # 28: channels per super-block (R0+R1 shared index)


BIAS_BYTE_OFFSET = 1  # super-block byte 0 is the bias; channels start at byte 1


# Mask slot assignment — a single R_MASK blob (loaded once at init) carries all
# three slots the asm needs.  Left/right edge columns are applied by mask_shift,
# NOT by slots; the slots only zero whole out-of-bounds rows:
#   0 = none        (KEEP all)             -> interior / kr=0-row taps
#   3 = top-row     (zero packed row 0)    -> g0 section kr=-1 taps
#   6 = bottom-row  (zero last packed row) -> gN section kr=+1 taps
# The g0 section selects slot 3, the gN section selects slot 6 — no reload.
MASK_SLOT_NONE = 0
MASK_SLOT_TOP = 3
MASK_SLOT_BOTTOM = 6


def build_border_mask_blob(cols: int) -> bytes:
    """Build the single 128-byte (8 x 16-byte slot) R_MASK blob.

    Mask polarity (matches upstream ``_mult_mask_and_shift``): a mask bit of
    **1 KEEPS** the lane, **0 ZEROES** it.  ``rows_per_chunk`` = 128 // cols
    spatial rows are packed into the 128 lanes; row ``r`` occupies lanes
    ``[r*cols, r*cols + cols)``.

    Left/right edge columns are handled at runtime by ``mask_shift`` (with
    ``CR15.partition = cols``), so the slots only zero whole out-of-bounds rows:

      slot 0 (none)       -> KEEP every lane (interior / kr=0-row taps)
      slot 3 (top row)    -> ZERO packed row 0       (g0 section kr=-1 taps)
      slot 6 (bottom row) -> ZERO the last packed row (gN section kr=+1 taps)

    One blob carries all three; the asm selects slot 3 in g0 and slot 6 in gN,
    so no mid-program R_MASK reload is needed.
    """
    rows_per_chunk = 128 // cols
    top_bits = set(range(0, cols))                                  # row 0
    bottom_row = rows_per_chunk - 1
    bottom_bits = set(range(bottom_row * cols, bottom_row * cols + cols))

    # Per slot, the set of lanes to ZERO (bit cleared); all others kept (bit 1).
    zero_lanes = {
        MASK_SLOT_NONE: set(),
        MASK_SLOT_TOP: top_bits,
        MASK_SLOT_BOTTOM: bottom_bits,
    }

    mask = bytearray(128)
    for slot, zeros in zero_lanes.items():
        for bit in range(128):
            if bit not in zeros:
                byte_idx = slot * 16 + bit // 8
                mask[byte_idx] |= 1 << (bit % 8)
    return bytes(mask)



def _pack_conv_weights_fpb28(
    weights_reordered: np.ndarray, dtype: DType, bias: np.ndarray
) -> bytes:
    """Pack [out_ch, in_ch, 9] (taps reordered) + per-filter bias into FPB=28 blocks.

    Byte 0 of every 256-byte super-block holds the filter's INT8 bias; channel
    ``s`` occupies bytes ``[1 + s*9 .. 1 + s*9 + 9)`` (uniform +1 offset). The
    bias byte is replicated into every super-block of a filter so the asm's
    weight index is the same across blocks; the asm only accumulates the bias
    once per filter (from block 0). 1 + 28*9 = 253 <= 256, capacity unchanged.

    Per-filter byte stride: ceil(in_ch / 28) * 256.
    """
    out_ch, in_ch, k2 = weights_reordered.shape
    if k2 != 9:
        raise ValueError(f"expected last dim=9 (taps), got {k2}")
    if bias.shape != (out_ch,):
        raise ValueError(f"bias must have shape ({out_ch},), got {bias.shape}")
    raw = cast_to_wire_bytes(weights_reordered, dtype)
    bias_bytes = cast_to_wire_bytes(bias, dtype)
    # raw indexing: byte for filter f, channel ic, tap t = raw[(f*in_ch+ic)*9 + t]

    super_blocks_per_filter = math.ceil(in_ch / FPB)
    total = out_ch * super_blocks_per_filter * SUPER_BLOCK_BYTES
    packed = bytearray(total)

    for f in range(out_ch):
        for sb in range(super_blocks_per_filter):
            sb_base = (f * super_blocks_per_filter + sb) * SUPER_BLOCK_BYTES
            packed[sb_base] = bias_bytes[f]   # byte 0 = filter bias
            for s in range(FPB):
                ic = sb * FPB + s
                if ic >= in_ch:
                    break
                src = (f * in_ch + ic) * 9
                dst = sb_base + BIAS_BYTE_OFFSET + s * 9   # 1,10,19,...
                packed[dst:dst + 9] = raw[src:src + 9]
    return bytes(packed)


class ConvUniversalBnActivationApp(IpuApp):
    """Universal 3x3 convolution + folded-bias + ReLU application harness.

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary (chunk-interleaved layout).
        kernel:       Numpy weights of shape ``[out_ch, in_ch, 3, 3]``.
        kernel_path:  Alternative: path to a raw ``[out_ch, in_ch, 9]``
                      contiguous byte file.
        bias:         Per-output-channel INT8 bias, shape ``[out_ch]``. Added
                      once to the accumulator before ReLU. Defaults to zeros.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height.
        cols:         Spatial width; one of {16, 32, 64} (one packed row per
                      mask partition group).
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
        bias: Optional[np.ndarray] = None,
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

        if bias is None:
            bias = np.zeros(out_channels, dtype=np.int8)
        bias = np.asarray(bias)
        if bias.shape != (out_channels,):
            raise ValueError(
                f"bias must have shape ({out_channels},), got {bias.shape}"
            )
        self._bias_array = bias

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
        self.in_group_stride = in_channels * CHUNK_BYTES
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
        return _pack_conv_weights_fpb28(w_reordered, self.dtype, self._bias_array)

    def setup(self, state: "IpuState") -> None:
        # Set data type (master ISA: dtype is a state attribute, not a CR register).
        state.dtype = self.dtype

        # Load input
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        # Pack and load kernel (dense FPB=28 super-block layout)
        kernel_packed = self._pack_kernel()
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_packed)

        # Border masks: a SINGLE blob carrying all 3 slots (0=none, 3=top-row
        # zero, 6=bottom-row zero), loaded once at init.  The g0 section selects
        # slot 3, the gN section selects slot 6 — no mid-program R_MASK reload.
        # Vertical out-of-bounds rows are masked (no zero chunk in the cyclic
        # register); left/right edge columns are applied at runtime by mask_shift
        # (see CR15 partition below).
        state.xmem.write_address(MASK_BASE_ADDR, build_border_mask_blob(self.cols))

        # CR15 dstructure: partition so each partition group is exactly one
        # packed spatial row (group size == cols).  The asm's mask_shift then
        # injects the left/right edge-column zero at each packed-row boundary.
        cols_to_partition = {
            64: Partition.P2,   # 2 groups of 64 lanes
            32: Partition.P4,   # 4 groups of 32 lanes
            16: Partition.P8,   # 8 groups of 16 lanes
        }
        if self.cols not in cols_to_partition:
            raise ValueError(
                f"conv_universal_bn_activation mask-shift scheme requires cols in "
                f"{sorted(cols_to_partition)} (one packed row per partition group); "
                f"got cols={self.cols}"
            )
        state.set_cr_dstructure(
            valid_elements=128,
            partition=cols_to_partition[self.cols],
        )

        # CR register map — adapted to master ISA:
        #   CR0 is read-only constant 0, CR1 is read-only constant 1, CR15 is the
        #   dstructure (valid_elements | partition). So the input/kernel bases that
        #   used to live in CR0/CR1 were relocated:
        #     CR10 = INPUT_BASE_ADDR   (was CR0; CR0 now serves the zero-constant role)
        #     CR5  = KERNEL_BASE_ADDR  (was CR1; CR5's old num_chunks value is unused in asm)
        state.regfile.set_cr(10, INPUT_BASE_ADDR)
        state.regfile.set_cr(5, KERNEL_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, MASK_BASE_ADDR)          # single mask blob (slots 0/3/6)
        # cr9 is no longer a mask base (no reload); left unset.

        # Set parameter CR registers
        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(6, self.in_group_stride)
        state.regfile.set_cr(7, FPB * CHUNK_BYTES)  # channel group size = 28 * 128 = 3584
        state.regfile.set_cr(8, self.total_kernel_bytes)
        # cr11 = chunk-loop limit = (num_chunks - 1) * in_group_stride
        # Used by asm to compare lr8 (chunk base addr) against chunk limit,
        # replacing the previous lr9 chunk counter.
        state.regfile.set_cr(11, (self.num_chunks - 1) * self.in_group_stride)

        # Constants used by the asm. CR0 is the read-only zero constant (master ISA),
        # which the asm now uses for "SET lr<n> cr0".
        state.regfile.set_cr(12, CHUNK_BYTES)        # 128
        state.regfile.set_cr(13, SUPER_BLOCK_BYTES)  # 256
        # cr14 = end-of-9 walking-pointer step: brings lr_walk from this ch's
        # tap-9 offset (lr_read + cols + 1) to next ch's tap-1 offset
        # ((lr_read + SUPER_BLOCK_BYTES) - cols - 1), i.e. +(SUPER_BLOCK_BYTES - 2*cols - 2).
        state.regfile.set_cr(14, (SUPER_BLOCK_BYTES - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.out_channels
            dump_outputs(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_outputs,
            )
