"""Unified pointwise (1x1) convolution harness (FP32 wide-vector mode).

A single code path that handles any valid pointwise convolution via a
multi-pass inner loop. Runs on the emulator's wide-vector debug datapath
(128 elements of 32-bit FP32 per vector, see
docs/content/wide-vector-debug-mode.md) -- weights, activations, and output
are all genuine floats, no INT8 quantization anywhere in this kernel.
Hardware behaviour is otherwise unchanged; masking, addressing, and the
pass structure are identical to the (retired) INT8 version.

Kernel layout: one OC per 128-element register-load, padded with zeros to
128 elements per pass. ``num_passes = ceil(in_channels / 128)``.

Constraints:
  - in_channels % 8 == 0  (avoids the runtime guard ever firing)
  - out_channels % 4 == 0
  - spatial: any height/width >= 1 -- padded internally to satisfy the
    hardware's ``cols`` divides 128 / whole-chunk constraints (see
    _spec_support.pointwise_pad_shape)

See DESIGN.md (sibling file) for cycle accounting and asm structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
# 512 bytes at FP32's 4 B/element. r_cyclic ELEMENT addressing (the `index`
# operand of LDR_CYCLIC_MULT_REG and MULT.RC.VE's `rc_idx`) is untouched by
# this migration -- see the .asm header for the full recipe note.

ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode
MASK_SLOT_BYTES = 16  # 128-bit mask, mode-blind (not widened)

class PointwiseConvUnifiedApp(IpuApp):
    """Unified pointwise (1x1) convolution application harness (FP32).

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor, matching
    the repo-wide rule that a caller's file layout is never leaked into by
    internal packing/padding (see docs/content/adding-applications.md):

      - ``input_path``:  raw ``[in_channels, height, width]`` float32 bytes,
        row-major (i.e. ``input.astype(np.float32).tobytes()``).
      - ``output_path``: raw ``[out_channels, height, width]`` float32 bytes,
        same convention -- padding this app applies internally (to satisfy
        the hardware's ``cols`` divides 128 / whole-chunk constraints) is
        truncated back off before the file is written.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary (see above).
        kernel_path:  Path to kernel binary, raw ``[out_channels, in_channels]``
                      float32.
        output_path:  Optional path to write output (see above).
        height:       Spatial height (>= 1). Any value works -- padded
                      internally to satisfy the hardware's chunk constraints.
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
        # Padded to satisfy the hardware's real constraints: `cols` must
        # divide CHUNK_ELEMENTS (128) (a spatial row must tile one chunk
        # without straddling its edge -- the packing loop writes
        # packed[dst + r*cols : dst + r*cols + cols]), and rows*cols must be
        # a whole number of chunks. Pointwise has no spatial neighbourhood,
        # so a padded lane can never leak into a real lane through the conv
        # -- see _spec_support.pointwise_pad_shape.
        self.rows, self.cols = pointwise_pad_shape(height, width)
        rows, cols = self.rows, self.cols

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
        # Size each region from THIS configuration instead of fixed gaps.
        # Region sizes are in ELEMENTS (CHUNK_ELEMENTS/row); setup() scales
        # to bytes via ROW_BYTES (FP32, 4 B/element, always).
        input_rows = self.row_groups * in_channels
        kernel_rows = out_channels * num_passes
        output_rows = self.row_groups * out_channels
        self._regions = allocate_regions([
            ("input", input_rows * CHUNK_ELEMENTS),
            ("kernel", kernel_rows * CHUNK_ELEMENTS),
            ("mask", CHUNK_ELEMENTS),
            ("output", output_rows * CHUNK_ELEMENTS),
        ])
        self.input_base_row = self._regions["input"] // CHUNK_ELEMENTS
        self.kernel_base_row = self._regions["kernel"] // CHUNK_ELEMENTS
        self.mask_base_row = self._regions["mask"] // CHUNK_ELEMENTS
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
                # ICs in this pass: 128 (full) or tail_size (last)
                ics_in_pass = 128 if p < P - 1 else self.tail_size
                dst_base = (oc * P + p) * CHUNK_ELEMENTS
                src_base = oc * in_ch + pass_start_ic
                packed[dst_base:dst_base + ics_in_pass] = (
                    raw_kernel_f32[src_base:src_base + ics_in_pass]
                )
                # elements [ics_in_pass..128) stay zero (padding)
        return packed.tobytes()

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires.

        ``wide_vector_quantize_output=False`` keeps elements 4-byte FP32
        through AAQ (AAQ is a no-op); ACTIVATE writes FP32 into
        POST_AAQ_REG and STR_POST_AAQ_REG drains the full 512 bytes -- same
        convention as softmax_rows.
        """
        return IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )

    def setup(self, state: "IpuState") -> None:
        # input_path holds the TRUE (unpadded) [in_channels, height, width]
        # float32 tensor -- pad + chunk-pack to the on-device layout here,
        # internal to this method (see the class docstring's file-layout
        # contract).
        input_raw = np.frombuffer(self.input_path.read_bytes(), dtype=np.float32)
        input_chw = input_raw.reshape(self.in_channels, self.height, self.width)
        padded = np.zeros((self.in_channels, self.rows, self.cols), dtype=np.float32)
        padded[:, :self.height, :self.width] = input_chw
        input_data = pack_input_chunked(padded, self.cols)
        state.xmem.write_address(self.input_base_row * ROW_BYTES, input_data)

        kernel_raw = np.frombuffer(self.kernel_path.read_bytes(), dtype=np.float32)
        kernel_packed = self._pack_kernel(kernel_raw)
        state.xmem.write_address(self.kernel_base_row * ROW_BYTES, kernel_packed)

        # Mask polarity (master, 2026-06-14): bit 1 = KEEP lane, bit 0 = ZERO.
        # This app never masks, so slot 0 must be all-ones (keep every lane).
        # The mask blob does NOT widen -- it is 1 bit per lane regardless of
        # arithmetic mode -- only its row address scales with ROW_BYTES.
        state.xmem.write_address(self.mask_base_row * ROW_BYTES, b"\xff" * MASK_SLOT_BYTES)

        # Master ISA: CR0 = read-only 0, CR1 = read-only 1 (cannot be overwritten).
        # INPUT_BASE_ROW is 0, so CR0 serves as both the zero constant and the
        # input/cyclic-load base.  The kernel base (nonzero) is relocated to CR14
        # (whose old role, the constant 1 pass decrement, now uses CR1 directly).
        # All of these are XMEM *row* numbers, not byte addresses -- the
        # emulator scales them by ROW_BYTES.
        state.regfile.set_cr(2, self.mask_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(14, self.kernel_base_row)

        # Parameter CR registers (see DESIGN.md). cr8/cr13 are XMEM-space
        # (rows); cr4/cr5/cr6/cr7/cr9/cr10/cr11 are plain counters/lane
        # bounds, unaffected by row addressing. CR scalars stay integer-only
        # even in wide mode (see docs/content/wide-vector-debug-mode.md) --
        # none of these carry fractional values, so this is unaffected by
        # the FP32 migration.
        state.regfile.set_cr(4, self.num_passes)
        state.regfile.set_cr(5, self.row_groups)
        state.regfile.set_cr(6, self.pipeline_limit_full)
        state.regfile.set_cr(7, self.out_channels)
        state.regfile.set_cr(8, self.row_group_stride)  # ROWS (= in_channels)
        # pipeline_limit_tail may be negative; encode as two's complement
        state.regfile.set_cr(9, self.pipeline_limit_tail & 0xFFFFFFFF)
        state.regfile.set_cr(10, self.tail_size)
        state.regfile.set_cr(11, self.num_passes - 1)

        # cr12 = 128: the ONE remaining role is the fixed_idx step (lane/
        # element space, mode-blind) for Half B -- NOT an XMEM stride anymore;
        # that role (kernel-row advance, output-row advance, output
        # pre-offset, per-IC walking step) moved to the read-only CR1 (= 1
        # row) throughout the .asm.
        state.regfile.set_cr(12, 128)
        state.regfile.set_cr(13, self.pass_stride_rows)  # 128 ROWS
        # (pass-counter decrement constant 1 = read-only CR1; CR14 holds the kernel base.)

        # Master ISA: ACTIVATE.QUANTIZE reads its active-lane count from the named
        # dstructure CR's valid_elements field. The asm names CR15, so set
        # CR15.valid_elements = 128 to activate the full 128-lane output chunk.
        # (Mults use mask_offset 0 / no masking, so partition is irrelevant.)
        # With wide_vector_quantize_output=False (make_state above), ACTIVATE
        # writes identity-activated FP32 into post_aaq_reg with no clamp --
        # this is NOT a quantize step in this mode; the name is a holdover
        # from the INT8 code path this app used to run.
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
        # Always run on the FP32 wide-vector state unless caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)

