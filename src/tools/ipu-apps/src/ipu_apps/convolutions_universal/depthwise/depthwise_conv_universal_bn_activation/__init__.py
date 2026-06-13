"""Universal depthwise 3x3 convolution + folded-bias + ReLU harness.

Derived from ``depthwise_conv_universal``. Same chunk-interleaved I/O layout and
walking-pointer / rotating-cyclic-slot pipeline, with three additions
(mirroring ``conv_universal_bn_activation``):

  * **Folded bias** — one INT8 bias per channel, injected as a single extra
    "multiply by 1" accumulate (``acc.first``) at the start of each channel.
    Batch-norm is assumed already folded into the depthwise weights + this bias.
  * **ReLU activation** — applied via ``ACTIVATE relu`` before quantization.
  * **Mask-based borders** — the top/bottom out-of-bounds rows are zeroed with
    partition-composite mask slots instead of loading a zero chunk into the
    cyclic register (no zero region).

Per-channel budget: **10 cyc/ch** = 1 bias-seed cycle + 9 weight taps (the base
app runs 9 cyc/ch with no bias). The deferred-store pipeline (store the previous
channel's quantized result while the current channel computes) is preserved.

Kernel super-block layout (FPB=25, stride 10):
  Depthwise produces one output PER channel, so each channel needs its OWN bias
  byte — conv's "one bias byte shared across a super-block's in-channels" does
  not apply. Instead each channel occupies a **10-byte slot**: byte 0 = its INT8
  bias, bytes 1..9 = its 9 weight taps. 25 channels * 10 = 250 <= 256, so one
  256-byte super-block (R0 = bytes 0..127, R1 = 128..255) holds 25 channels.
  The shared ``mult.ve`` fixed_idx (0..255) addresses both halves transparently.

  The asm walks one continuous kernel byte index ``lr6`` at +1 per cycle: for
  channel ``s`` the bias-seed reads ``fixed_idx = s*10`` (bias), then the 9 taps
  read ``s*10 + 1 .. s*10 + 9``; the next channel's bias is the following byte,
  so the 10-cycle/channel body advances ``lr6`` by exactly one channel stride.

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal_bn_activation import (
        DepthwiseConvUniversalBnActivationApp,
    )

    app = DepthwiseConvUniversalBnActivationApp(
        inst_path="depthwise_conv_universal_bn_activation.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",   # channels * 9 bytes, INT8 raw
        bias=bias,                  # np.ndarray [channels], INT8 (defaults zeros)
        output_path="output.bin",
        dtype="INT8",
        rows=64, cols=64, channels=256,
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
    dump_outputs,
)
# Reuse the conv_universal_bn_activation mask-blob builder so the two apps share
# one border-mask implementation.
from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
    build_border_mask_blob,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x110000
# Border handling is done entirely with masks (no zero region). Two 128-byte
# mask blobs: TOP composites (chunk-0 / g0 section) and BOTTOM composites
# (reloaded into R_MASK for the last-chunk / gN section).
MASK_TOP_BASE_ADDR = 0x120000
MASK_BOTTOM_BASE_ADDR = 0x120080
OUTPUT_BASE_ADDR = 0x130000

OUTPUT_CHUNK_BYTES = 128  # 128 bytes per output channel per chunk (int8)

FPB = 25            # channels per 256-byte super-block (1 bias + 9 taps each)
CH_SLOT_BYTES = 10  # per-channel slot: byte 0 = bias, bytes 1..9 = 9 taps
SUPER_BLOCK_BYTES = 256


def _pack_depthwise_kernel_bias(
    kernel_raw: bytes, bias_bytes: bytes, channels: int
) -> bytes:
    """Pack per-channel (bias + 9 weight taps) into FPB=25 super-blocks.

    Input:  ``kernel_raw`` = channels*9 bytes (channel ch's 9 taps at ch*9);
            ``bias_bytes`` = channels bytes (channel ch's INT8 bias).
    Output: ceil(channels/25) super-blocks of 256 bytes each.

    Within one super-block, channel ``s`` (0..24) occupies bytes
    ``[s*10 .. s*10 + 10)``: byte ``s*10`` = bias, ``s*10+1 .. s*10+9`` = taps.
    25*10 = 250 <= 256.  R0 holds bytes 0..127, R1 holds 128..255; the
    shared-index ``mult.ve`` (fixed_idx 0..255) spans both halves.
    """
    num_blocks = math.ceil(channels / FPB)
    total = num_blocks * SUPER_BLOCK_BYTES
    packed = bytearray(total)
    for sb in range(num_blocks):
        sb_base = sb * SUPER_BLOCK_BYTES
        for s in range(FPB):
            ch = sb * FPB + s
            if ch >= channels:
                break
            slot = sb_base + s * CH_SLOT_BYTES
            packed[slot] = bias_bytes[ch]                       # byte 0 = bias
            packed[slot + 1:slot + 10] = kernel_raw[ch * 9:ch * 9 + 9]
    return bytes(packed)


class DepthwiseConvUniversalBnActivationApp(IpuApp):
    """Universal depthwise 3x3 convolution + folded-bias + ReLU harness.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary (chunk-interleaved layout).
        kernel_path:  Path to kernel binary (channels * 9 bytes, INT8 raw).
        bias:         Per-channel INT8 bias, shape ``[channels]``. Added once to
                      the accumulator before ReLU. Defaults to zeros.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height.
        cols:         Spatial width (16, 32, or 64).
        channels:     Number of channels (>= 1).
    """

    def __init__(
        self,
        *,
        dtype: str | DType = "INT8",
        rows: int,
        cols: int,
        channels: int,
        bias: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

        valid_cols = {16, 32, 64}
        if cols not in valid_cols:
            raise ValueError(f"cols must be in {valid_cols}, got {cols}")
        num_chunks = (rows * cols) // 128
        if num_chunks < 2:
            raise ValueError(
                f"Need at least 2 chunks (rows*cols >= 256), got {rows}*{cols}={rows*cols}"
            )
        if channels < 1:
            raise ValueError(f"channels ({channels}) must be >= 1")

        if bias is None:
            bias = np.zeros(channels, dtype=np.int8)
        bias = np.asarray(bias)
        if bias.shape != (channels,):
            raise ValueError(f"bias must have shape ({channels},), got {bias.shape}")
        self._bias_array = bias

        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.num_chunks = num_chunks
        self.group_stride = channels * 128
        self.num_super_blocks = math.ceil(channels / FPB)
        self.total_kernel_bytes = self.num_super_blocks * SUPER_BLOCK_BYTES

    def setup(self, state: "IpuState") -> None:
        # Master ISA: dtype is a state attribute, not a CR register.
        state.dtype = self.dtype

        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        kernel_raw = self.kernel_path.read_bytes()
        expected = self.channels * 9
        if len(kernel_raw) != expected:
            raise ValueError(
                f"kernel_path file has {len(kernel_raw)} bytes, "
                f"expected {expected} (channels * 9)"
            )
        bias_bytes = self._bias_array.astype(np.int8).view(np.uint8).tobytes()
        kernel_packed = _pack_depthwise_kernel_bias(
            kernel_raw, bias_bytes, self.channels
        )
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_packed)

        # Border masks: TOP blob for the g0 section, BOTTOM blob reloaded into
        # R_MASK for the gN section. No zero region.
        state.xmem.write_address(MASK_TOP_BASE_ADDR, build_border_mask_blob(self.cols, "top"))
        state.xmem.write_address(MASK_BOTTOM_BASE_ADDR, build_border_mask_blob(self.cols, "bottom"))

        # CR map (master ISA: CR0 = read-only 0, CR1 = read-only 1, CR15 = dstructure).
        # Relocate the input/kernel bases off CR0/CR1 (mirroring
        # conv_universal_bn_activation), keeping CR0 free as the read-only zero
        # constant used by "SET lr<n>, cr0":
        #   CR10 = INPUT_BASE (cyclic-load base), CR5 = KERNEL_BASE,
        #   CR3 = TOP mask blob, CR9 = BOTTOM mask blob (gN reload base).
        state.regfile.set_cr(10, INPUT_BASE_ADDR)
        state.regfile.set_cr(5, KERNEL_BASE_ADDR)
        # cr2 is pre-biased by -128 for the deferred store (asm advances lr7
        # BEFORE the XMEM store at tap 2; store writes to lr7_advanced + cr2 =
        # lr7_old + OUTPUT_BASE_ADDR).
        state.regfile.set_cr(2, (OUTPUT_BASE_ADDR - 128) & 0xFFFFFFFF)
        state.regfile.set_cr(3, MASK_TOP_BASE_ADDR)
        state.regfile.set_cr(9, MASK_BOTTOM_BASE_ADDR)

        # Parameter CR registers
        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(6, self.group_stride)
        state.regfile.set_cr(7, FPB * 128)         # channel group inner-loop size in bytes
        state.regfile.set_cr(8, self.total_kernel_bytes)
        state.regfile.set_cr(11, (self.num_chunks - 1) * self.group_stride)
        state.regfile.set_cr(12, 128)
        state.regfile.set_cr(13, 256)
        state.regfile.set_cr(14, (256 - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.channels
            dump_outputs(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_outputs,
            )
