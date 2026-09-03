"""Unified pointwise (1x1) convolution + folded BN bias + ReLU harness (FP32).

BN/activation twin of ``pointwise_conv_unified``: same single multi-pass code
path, on the FP32 wide-vector debug datapath (see
docs/content/wide-vector-debug-mode.md), with two additions:

  * **Folded bias** — one float32 bias per output channel, seeded into the
    accumulator (``r_acc = bias``) once per OC before the conv taps, via a
    ``MULT.EE`` broadcast of the bias element (x CR1 = 1). Batch-norm is
    assumed already folded into the conv weights + this bias.
  * **ReLU** — applied via ``ACTIVATE relu`` (instead of identity).

Kernel layout: one OC per 128-element register-load, padded with zeros to
128 elements per pass. ``num_passes = ceil(in_channels / 128)``.

Bias layout: the bias region *mirrors the kernel* — ``out_ch × num_passes ×
128`` elements, one 128-element block per (OC, pass), with the OC's float32
bias in element 0 of its **pass-0** block (every other element is zero).
This lets the asm reuse ``lr12`` (the kernel element offset) to index the
bias region verbatim, via ``cr10`` instead of ``cr14`` — no extra pointer or
arithmetic. Only pass-0 blocks are ever read; the pass-1+ blocks are pure
padding (wasteful for multi-pass, but bias regions are KB-scale against 2 MB
of XMEM).

(CR15 is reserved and rejected as an ISA operand, and cr0..cr14 are all
assigned by the base app; the base app's cr10 — a vestigial ``tail_size``
param never read as an operand — is reused here for the bias base.)

Constraints:
  - in_channels % 8 == 0  (avoids the runtime guard ever firing)
  - out_channels % 4 == 0
  - spatial: any height/width >= 1 -- padded internally (see
    _spec_support.pointwise_pad_shape)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    dump_outputs,
    allocate_regions,
    pack_input_chunked,
    unpack_output_chunked,
    CHUNK_ELEMENTS,
)
from ipu_apps.convolutions_universal._spec_support import (
    pointwise_pad_shape,
)

if TYPE_CHECKING:
    pass

# -- Memory layout -----------------------------------------------------------
#
# Row-addressed ISA (mb/195): XMEM offset/base operands on LDR_MULT_REG /
# LDR_CYCLIC_MULT_REG's offset+base / LDR_MULT_MASK_REG / STR_POST_AAQ_REG are
# ROW numbers, not byte addresses. A "row" is CHUNK_ELEMENTS (128) elements --
# 512 bytes at FP32's 4 B/element -- including BIAS_BASE_ROW, which feeds cr10
# exactly like KERNEL_BASE_ROW feeds cr14 (same LDR_MULT_REG pattern, same
# row-number treatment). r_cyclic ELEMENT addressing is untouched -- see the
# .asm header for the full recipe note.

ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode
MASK_SLOT_BYTES = 16  # 128-bit mask, mode-blind (not widened)

class PointwiseConvUnifiedBnActivationApp(IpuApp):
    """Unified pointwise (1x1) convolution + folded-bias + ReLU harness (FP32).

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor -- see
    the sibling ``pointwise_conv_unified``'s class docstring for the exact
    file-layout contract; it is identical here.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary, raw ``[in_channels, height,
                      width]`` float32.
        kernel_path:  Path to kernel binary, raw ``[out_channels, in_channels]``
                      float32.
        bias:         Per-output-channel float32 bias, shape ``[out_channels]``.
                      Added once to the accumulator before ReLU. Defaults to
                      zeros.
        output_path:  Optional path to write output (raw ``[out_channels,
                      height, width]`` float32).
        height:       Spatial height (>= 1). Any value works -- padded
                      internally.
        width:        Spatial width (>= 1). Any value works, same as height.
        in_channels:  Number of input channels (multiple of 8).
        out_channels: Number of output channels (multiple of 4).
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        in_channels: int,
        out_channels: int,
        bias: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)

        if in_channels % 8 != 0 or in_channels < 8:
            raise ValueError(
                f"in_channels ({in_channels}) must be a positive multiple of 8"
            )
        if out_channels % 4 != 0 or out_channels < 4:
            raise ValueError(
                f"out_channels ({out_channels}) must be a positive multiple of 4"
            )
        if height < 1:
            raise ValueError(f"height must be >= 1, got {height}")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")

        self.height = height
        self.width = width
        self.rows, self.cols = pointwise_pad_shape(height, width)
        rows, cols = self.rows, self.cols

        if bias is None:
            bias = np.zeros(out_channels, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape != (out_channels,):
            raise ValueError(
                f"bias must have shape ({out_channels},), got {bias.shape}"
            )
        self._bias_array = bias

        # Derive multi-pass parameters
        num_passes = (in_channels + 127) // 128
        # tail_size: ICs handled by the LAST pass (1..128)
        tail_size = in_channels - (num_passes - 1) * 128

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_passes = num_passes
        self.tail_size = tail_size

        # Derived constants. row_group_stride/pass_stride are ROW-granular
        # (XMEM-space): one input channel occupies one 128-element XMEM row,
        # so "in_channels * 128 elements" becomes "in_channels rows" and the
        # 128-IC pass stride becomes 128 rows.
        self.rows_per_chunk = CHUNK_ELEMENTS // cols
        self.row_groups = (rows * cols) // CHUNK_ELEMENTS
        self.row_group_stride = in_channels  # rows
        self.pass_stride_rows = 128  # rows (one pass = 128 input channels)

        # pipeline_limit for full 128-IC passes: 128 - 5 = 123
        # pipeline_limit for tail pass: tail_size - 5 (may be negative)
        self.pipeline_limit_full = 128 - 5
        self.pipeline_limit_tail = tail_size - 5

        # -- Dynamic region layout -------------------------------------------
        # See the sibling pointwise_conv_unified for the full rationale: the
        # fixed 64 KiB kernel gap silently overflowed into the next region
        # once out_channels * num_passes * 128 exceeded it. The bias region
        # mirrors the kernel's block grid exactly (see _pack_bias), so it
        # overflows on precisely the same configurations and is sized the
        # same way here. Region sizes are in ELEMENTS; setup() scales to
        # bytes via ROW_BYTES (FP32, 4 B/element, always).
        input_rows = self.row_groups * in_channels
        kernel_rows = out_channels * num_passes
        output_rows = self.row_groups * out_channels
        self._regions = allocate_regions([
            ("input", input_rows * CHUNK_ELEMENTS),
            ("kernel", kernel_rows * CHUNK_ELEMENTS),
            ("mask", CHUNK_ELEMENTS),
            ("bias", kernel_rows * CHUNK_ELEMENTS),
            ("output", output_rows * CHUNK_ELEMENTS),
        ])
        self.input_base_row = self._regions["input"] // CHUNK_ELEMENTS
        self.kernel_base_row = self._regions["kernel"] // CHUNK_ELEMENTS
        self.mask_base_row = self._regions["mask"] // CHUNK_ELEMENTS
        self.bias_base_row = self._regions["bias"] // CHUNK_ELEMENTS
        self.output_base_row = self._regions["output"] // CHUNK_ELEMENTS
        self.output_base_addr = self._regions["output"] * 4  # bytes, FP32

    def _pack_kernel(self, raw_kernel_f32: np.ndarray) -> bytes:
        """Pack kernel with oc_per_reg=1 layout, zero-padded, as FP32.

        Raw layout: raw_kernel_f32[oc * in_channels + ic]
        Packed layout:
          [OC 0, pass 0: 128 elements]
          [OC 0, pass 1: 128 elements]
          ...
          [OC 0, pass P-1: tail padded to 128]
          [OC 1, pass 0: ...]
          ...

        Total = out_channels * num_passes * 128 elements (float32).
        """
        P = self.num_passes
        in_ch = self.in_channels
        out_ch = self.out_channels

        packed = np.zeros(out_ch * P * CHUNK_ELEMENTS, dtype=np.float32)
        for oc in range(out_ch):
            for p in range(P):
                pass_start_ic = p * 128
                ics_in_pass = 128 if p < P - 1 else self.tail_size
                dst_base = (oc * P + p) * CHUNK_ELEMENTS
                src_base = oc * in_ch + pass_start_ic
                packed[dst_base:dst_base + ics_in_pass] = (
                    raw_kernel_f32[src_base:src_base + ics_in_pass]
                )
        return packed.tobytes()

    def _pack_bias(self) -> bytes:
        """Pack per-OC float32 bias into a kernel-mirroring region.

        Region shape = out_ch x num_passes x 128 elements (identical block
        grid to the packed kernel -- see _pack_kernel). The OC's bias goes
        in element 0 of its **pass-0** block; every other element is zero.
        The asm indexes this with ``lr12`` (the kernel row offset, which
        sits at the OC's pass-0 block at OC entry) via cr10, so no separate
        pointer is needed.
        """
        P = self.num_passes
        out_ch = self.out_channels

        packed = np.zeros(out_ch * P * CHUNK_ELEMENTS, dtype=np.float32)
        for oc in range(out_ch):
            packed[(oc * P) * CHUNK_ELEMENTS] = self._bias_array[oc]
        return packed.tobytes()

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires (see
        pointwise_conv_unified.make_state for the same convention)."""
        return IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )

    def setup(self, state: "IpuState") -> None:
        input_raw = np.frombuffer(self.input_path.read_bytes(), dtype=np.float32)
        input_chw = input_raw.reshape(self.in_channels, self.height, self.width)
        padded = np.zeros((self.in_channels, self.rows, self.cols), dtype=np.float32)
        padded[:, :self.height, :self.width] = input_chw
        input_data = pack_input_chunked(padded, self.cols)
        state.xmem.write_address(self.input_base_row * ROW_BYTES, input_data)

        kernel_raw = np.frombuffer(self.kernel_path.read_bytes(), dtype=np.float32)
        kernel_packed = self._pack_kernel(kernel_raw)
        state.xmem.write_address(self.kernel_base_row * ROW_BYTES, kernel_packed)

        # Folded-bias region (mirrors the kernel layout — see _pack_bias).
        state.xmem.write_address(self.bias_base_row * ROW_BYTES, self._pack_bias())

        # Mask polarity (master, 2026-06-14): bit 1 = KEEP lane, bit 0 = ZERO.
        # This app never masks, so slot 0 must be all-ones (keep every lane).
        state.xmem.write_address(self.mask_base_row * ROW_BYTES, b"\xff" * MASK_SLOT_BYTES)

        # Master ISA: CR0 = read-only 0, CR1 = read-only 1 (cannot be overwritten).
        # INPUT_BASE_ROW is 0, so CR0 serves as both the zero constant and the
        # input/cyclic-load base.  The kernel base (nonzero) is relocated to CR14
        # (whose old role, the constant 1 pass decrement, now uses CR1 directly).
        # All of these are XMEM *row* numbers, not byte addresses.
        state.regfile.set_cr(2, self.mask_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(14, self.kernel_base_row)

        # Parameter CR registers (see DESIGN.md). CR scalars stay
        # integer-only even in wide mode -- none of these carry fractional
        # values, so unaffected by the FP32 migration.
        state.regfile.set_cr(4, self.num_passes)
        state.regfile.set_cr(5, self.row_groups)
        state.regfile.set_cr(6, self.pipeline_limit_full)
        state.regfile.set_cr(7, self.out_channels)
        state.regfile.set_cr(8, self.row_group_stride)  # ROWS (= in_channels)
        # pipeline_limit_tail may be negative; encode as two's complement
        state.regfile.set_cr(9, self.pipeline_limit_tail & 0xFFFFFFFF)
        # cr10: bias base ROW.  CR15 is reserved/illegal as an operand, and
        # all of cr0..cr14 are taken — but the base app's cr10 ("tail_size") is
        # never read as an operand, so it is reused here for the bias base.
        state.regfile.set_cr(10, self.bias_base_row)
        state.regfile.set_cr(11, self.num_passes - 1)

        # cr12 = 128: the ONE remaining role is the fixed_idx/ra_idx step
        # (lane/element space, mode-blind) for Half B -- NOT an XMEM stride;
        # that role moved to CR1 throughout the .asm.
        state.regfile.set_cr(12, 128)
        state.regfile.set_cr(13, self.pass_stride_rows)  # 128 ROWS
        # (pass-counter decrement constant 1 = read-only CR1; CR14 holds the kernel base.)

        # Master ISA: ACTIVATE reads its active-lane count from the named
        # dstructure CR's valid_elements field. The asm names CR15, so set
        # CR15.valid_elements = 128 to activate the full 128-lane chunk.
        # (Mults use mask_offset 0 / no masking, so partition is irrelevant; cr10
        # holds the bias base, cr15 is otherwise free for the dstructure config.)
        # With wide_vector_quantize_output=False, ACTIVATE (relu here, not
        # identity) writes FP32 into post_aaq_reg with no INT8 clamp.
        state.set_cr_dstructure(valid_elements=128)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_rows = self.row_groups * self.out_channels
            raw = state.xmem.read_address(
                self.output_base_addr, total_rows * ROW_BYTES
            )
            padded_out = unpack_output_chunked(raw, self.out_channels, self.rows, self.cols)
            out = padded_out[:, :self.height, :self.width]
            self.output_path.write_bytes(out.astype(np.float32).tobytes())

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)

