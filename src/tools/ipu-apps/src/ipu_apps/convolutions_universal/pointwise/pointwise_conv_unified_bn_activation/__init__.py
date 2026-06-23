"""Unified pointwise (1x1) convolution + folded BN bias + ReLU harness.

BN/activation twin of ``pointwise_conv_unified``: same single multi-pass code
path, with two additions:

  * **Folded bias** — one INT8 bias per output channel, seeded into the
    accumulator (``r_acc = bias``) once per OC before the conv taps, via a
    ``MULT.EE`` broadcast of the bias byte (× CR1 = 1). Batch-norm is assumed
    already folded into the conv weights + this bias.
  * **ReLU** — applied via ``ACTIVATE relu`` (instead of identity) before the
    INT8 quantize.

Kernel layout: one OC per 128-byte register-load, padded with zeros to
128 bytes per pass. ``num_passes = ceil(in_channels / 128)``.

Bias layout: the bias region *mirrors the kernel* — ``out_ch × num_passes ×
128`` bytes, one 128-byte block per (OC, pass), with the OC's INT8 bias in
byte 0 of its **pass-0** block (every other byte is zero). This lets the asm
reuse ``lr12`` (the kernel byte offset) to index the bias region verbatim, via
``cr10`` instead of ``cr14`` — no extra pointer or arithmetic. Only pass-0
blocks are ever read; the pass-1+ blocks are pure padding (wasteful for
multi-pass, but bias regions are KB-scale against 2 MB of XMEM).

(CR15 is reserved and rejected as an ISA operand, and cr0..cr14 are all
assigned by the base app; the base app's cr10 — a vestigial ``tail_size``
param never read as an operand — is reused here for the bias base.)

Constraints:
  - in_channels % 8 == 0  (avoids the runtime guard ever firing)
  - out_channels % 4 == 0
  - spatial: rows, cols power-of-2 in [16..128], rows*cols % 128 == 0
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_math import DType

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    parse_dtype,
    dump_outputs,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Memory layout -----------------------------------------------------------

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x110000
MASK_BASE_ADDR = 0x120000
BIAS_BASE_ADDR = 0x130000
# Output can span up to row_groups * out_ch * 128 bytes; for the largest config
# (64x64, oc=64) that is 0x40000, so OUTPUT must sit clear of the bias region.
OUTPUT_BASE_ADDR = 0x140000

OUTPUT_ROW_BYTES = 128


class PointwiseConvUnifiedBnActivationApp(IpuApp):
    """Unified pointwise (1x1) convolution + folded-bias + ReLU harness.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary.
        kernel_path:  Path to kernel binary.
        bias:         Per-output-channel INT8 bias, shape ``[out_channels]``.
                      Added once to the accumulator before ReLU. Defaults to
                      zeros.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height (power of 2, 16-128).
        cols:         Spatial width (power of 2, 16-128).
        in_channels:  Number of input channels (multiple of 8).
        out_channels: Number of output channels (multiple of 4).
    """

    def __init__(
        self,
        *,
        dtype: str | DType = "INT8",
        rows: int,
        cols: int,
        in_channels: int,
        out_channels: int,
        bias: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

        if in_channels % 8 != 0 or in_channels < 8:
            raise ValueError(
                f"in_channels ({in_channels}) must be a positive multiple of 8"
            )
        if out_channels % 4 != 0 or out_channels < 4:
            raise ValueError(
                f"out_channels ({out_channels}) must be a positive multiple of 4"
            )

        valid_spatial = {16, 32, 64, 128}
        if rows not in valid_spatial:
            raise ValueError(f"rows must be a power of 2 in {valid_spatial}, got {rows}")
        if cols not in valid_spatial:
            raise ValueError(f"cols must be a power of 2 in {valid_spatial}, got {cols}")

        if bias is None:
            bias = np.zeros(out_channels, dtype=np.int8)
        bias = np.asarray(bias)
        if bias.shape != (out_channels,):
            raise ValueError(
                f"bias must have shape ({out_channels},), got {bias.shape}"
            )
        self._bias_array = bias

        # Derive multi-pass parameters
        num_passes = (in_channels + 127) // 128
        # tail_size: ICs handled by the LAST pass (1..128)
        tail_size = in_channels - (num_passes - 1) * 128

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_passes = num_passes
        self.tail_size = tail_size

        # Derived constants
        self.rows_per_chunk = 128 // cols
        self.row_groups = (rows * cols) // 128
        self.row_group_stride = in_channels * 128

        # pipeline_limit for full 128-IC passes: 128 - 5 = 123
        # pipeline_limit for tail pass: tail_size - 5 (may be negative)
        self.pipeline_limit_full = 128 - 5
        self.pipeline_limit_tail = tail_size - 5

    def _pack_kernel(self, raw_kernel: bytes) -> bytes:
        """Pack kernel with oc_per_reg=1 layout, zero-padded.

        Raw layout: raw_kernel[oc * in_channels + ic]
        Packed layout:
          [OC 0, pass 0: 128 bytes]
          [OC 0, pass 1: 128 bytes]
          ...
          [OC 0, pass P-1: tail padded to 128]
          [OC 1, pass 0: ...]
          ...

        Total = out_channels * num_passes * 128 bytes.
        """
        P = self.num_passes
        in_ch = self.in_channels
        out_ch = self.out_channels

        # Pad out_channels up to even (we pair r0+r1). out_ch % 4 == 0 → already even.
        packed = bytearray(out_ch * P * 128)
        for oc in range(out_ch):
            for p in range(P):
                pass_start_ic = p * 128
                # ICs in this pass: 128 (full) or tail_size (last)
                ics_in_pass = 128 if p < P - 1 else self.tail_size
                dst_base = (oc * P + p) * 128
                src_base = oc * in_ch + pass_start_ic
                for i in range(ics_in_pass):
                    packed[dst_base + i] = raw_kernel[src_base + i]
                # bytes [ics_in_pass..128) stay zero (padding)
        return bytes(packed)

    def _pack_bias(self) -> bytes:
        """Pack per-OC INT8 bias into a kernel-mirroring region.

        Region shape = out_ch × num_passes × 128 (identical block grid to the
        packed kernel). The OC's bias goes in byte 0 of its **pass-0** block;
        every other byte is zero. The asm indexes this with ``lr12`` (the
        kernel byte offset, which sits at the OC's pass-0 block at OC entry)
        via cr15, so no separate pointer is needed.
        """
        P = self.num_passes
        out_ch = self.out_channels
        bias_bytes = self._bias_array.astype(np.int8).view(np.uint8)

        packed = bytearray(out_ch * P * 128)
        for oc in range(out_ch):
            # byte 0 of OC's pass-0 block (== same offset as kernel pass-0 block)
            packed[(oc * P) * 128] = int(bias_bytes[oc])
        return bytes(packed)

    def setup(self, state: "IpuState") -> None:
        # Master ISA: dtype is a state attribute, not a CR register.
        state.dtype = self.dtype

        input_data = self.input_path.read_bytes()
        state.xmem.write_address(INPUT_BASE_ADDR, input_data)

        kernel_raw = self.kernel_path.read_bytes()
        kernel_packed = self._pack_kernel(kernel_raw)
        state.xmem.write_address(KERNEL_BASE_ADDR, kernel_packed)

        # Folded-bias region (mirrors the kernel layout — see _pack_bias).
        state.xmem.write_address(BIAS_BASE_ADDR, self._pack_bias())

        # Mask polarity (master, 2026-06-14): bit 1 = KEEP lane, bit 0 = ZERO.
        # This app never masks, so slot 0 must be all-ones (keep every lane).
        state.xmem.write_address(MASK_BASE_ADDR, b"\xff" * 128)

        # Master ISA: CR0 = read-only 0, CR1 = read-only 1 (cannot be overwritten).
        # INPUT_BASE_ADDR is 0, so CR0 serves as both the zero constant and the
        # input/cyclic-load base.  The kernel base (nonzero) is relocated to CR14
        # (whose old role, the constant 1 pass decrement, now uses CR1 directly).
        state.regfile.set_cr(2, MASK_BASE_ADDR)
        state.regfile.set_cr(3, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(14, KERNEL_BASE_ADDR)

        # Parameter CR registers (see DESIGN.md)
        state.regfile.set_cr(4, self.num_passes)
        state.regfile.set_cr(5, self.row_groups)
        state.regfile.set_cr(6, self.pipeline_limit_full)
        state.regfile.set_cr(7, self.out_channels)
        state.regfile.set_cr(8, self.row_group_stride)
        # pipeline_limit_tail may be negative; encode as two's complement
        state.regfile.set_cr(9, self.pipeline_limit_tail & 0xFFFFFFFF)
        # cr10: bias base address.  CR15 is reserved/illegal as an operand, and
        # all of cr0..cr14 are taken — but the base app's cr10 ("tail_size") is
        # never read as an operand, so it is reused here for the bias base.
        state.regfile.set_cr(10, BIAS_BASE_ADDR)
        state.regfile.set_cr(11, self.num_passes - 1)

        # Constants
        state.regfile.set_cr(12, 128)
        state.regfile.set_cr(13, 16384)  # input pass stride: 128 ICs * 128B
        # (pass-counter decrement constant 1 = read-only CR1; CR14 holds the kernel base.)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_rows = self.row_groups * self.out_channels
            dump_outputs(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_ROW_BYTES, total_rows,
            )
