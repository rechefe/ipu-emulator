"""Unified pointwise (1x1) convolution harness.

A single code path that handles any valid pointwise convolution via a
multi-pass inner loop. Replaces the dual original/grouped paths of
``pointwise_conv_universal`` with one structure.

Kernel layout: one OC per 128-byte register-load, padded with zeros to
128 bytes per pass. ``num_passes = ceil(in_channels / 128)``.

Constraints:
  - in_channels % 8 == 0  (avoids the runtime guard ever firing)
  - out_channels % 4 == 0
  - spatial: cols divides 128 (power of 2, 1..128); rows >= 1;
    rows*cols % 128 == 0 (a whole number of 128-byte chunks)

See DESIGN.md (sibling file) for cycle accounting and asm structure.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType

import numpy as np

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    parse_dtype,
    dump_outputs,
    allocate_regions,
    pack_input_chunked,
    unpack_output_chunked,
)
from ipu_apps.convolutions_universal._spec_support import (
    REQUIRES,
    bias_requires_relu,
    conv_query,
    pointwise_pad_shape,
    positive_dims,
)
from ipu_apps.kernel_registry import KernelSpec, no, yes

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Memory layout -----------------------------------------------------------
#
# Row-addressed ISA (mb/195): XMEM offset/base operands on LDR_MULT_REG /
# LDR_CYCLIC_MULT_REG's offset+base / LDR_MULT_MASK_REG / STR_POST_AAQ_REG are
# ROW numbers, not byte addresses. *_BASE_ADDR below stay as byte constants
# for host-side xmem pokes (write_address/dump_outputs are byte-granular);
# *_BASE_ROW = *_BASE_ADDR // CHUNK_BYTES feeds the CR registers the asm
# actually loads/stores through. r_cyclic ELEMENT addressing (the `index`
# operand of LDR_CYCLIC_MULT_REG and MULT.RC.VE's `rc_idx`) is untouched by
# this migration -- see the .asm header for the full recipe note.

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x110000
MASK_BASE_ADDR = 0x120000
OUTPUT_BASE_ADDR = 0x130000

OUTPUT_ROW_BYTES = 128
CHUNK_BYTES = 128

INPUT_BASE_ROW = INPUT_BASE_ADDR // CHUNK_BYTES
KERNEL_BASE_ROW = KERNEL_BASE_ADDR // CHUNK_BYTES
MASK_BASE_ROW = MASK_BASE_ADDR // CHUNK_BYTES
OUTPUT_BASE_ROW = OUTPUT_BASE_ADDR // CHUNK_BYTES


def _as_signed_byte(value: int) -> int:
    """Reinterpret a wire byte as the signed INT8 it encodes."""
    v = value & 0xFF
    return v - 256 if v > 127 else v


class PointwiseConvUnifiedApp(IpuApp):
    """Unified pointwise (1x1) convolution application harness.

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor, matching
    the repo-wide rule that a caller's file layout is never leaked into by
    internal packing/padding (see docs/content/adding-applications.md):

      - ``input_path``:  raw ``[in_channels, height, width]`` INT8 bytes,
        row-major (i.e. ``input.astype(np.int8).tobytes()``).
      - ``output_path``: raw ``[out_channels, height, width]`` INT8 bytes,
        same convention -- padding this app applies internally (to satisfy
        the hardware's ``cols`` divides 128 / whole-chunk constraints) is
        truncated back off before the file is written.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary (see above).
        kernel_path:  Path to kernel binary, raw ``[out_channels, in_channels]``.
        output_path:  Optional path to write output (see above).
        dtype:        Data type string or :class:`DType`.
        height:       Spatial height (>= 1). Any value works -- padded
                      internally to satisfy the hardware's chunk constraints.
        width:        Spatial width (>= 1). Any value works, same as height.
        in_channels:  Number of input channels (multiple of 8).
        out_channels: Number of output channels (multiple of 4).
    """

    def __init__(
        self,
        *,
        dtype: str | DType = "INT8",
        height: int,
        width: int,
        in_channels: int,
        out_channels: int,
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
        if height < 1:
            raise ValueError(f"height must be >= 1, got {height}")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")

        self.height = height
        self.width = width
        # Padded to satisfy the hardware's real constraints: `cols` must
        # divide 128 (a spatial row must tile a 128-byte chunk without
        # straddling its edge -- the packing loop writes
        # packed[dst + r*cols : dst + r*cols + cols]), and rows*cols must be
        # a whole number of 128-byte chunks. Pointwise has no spatial
        # neighbourhood, so a padded lane can never leak into a real lane
        # through the conv -- see _spec_support.pointwise_pad_shape.
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
        # (XMEM-space): one input channel occupies one 128-byte XMEM row, so
        # "in_channels * 128 bytes" becomes "in_channels rows" and the
        # 128-IC pass stride becomes 128 rows.
        self.rows_per_chunk = 128 // cols
        self.row_groups = (rows * cols) // 128
        self.row_group_stride = in_channels  # rows
        self.pass_stride_rows = 128  # rows (one pass = 128 input channels)

        # pipeline_limit for full 128-IC passes: 128 - 5 = 123
        # pipeline_limit for tail pass: tail_size - 5 (may be negative)
        self.pipeline_limit_full = 128 - 5
        self.pipeline_limit_tail = tail_size - 5

        # Narrow-mode default; setup() overrides it once the state's mode is known.
        self._element_width = 1

        # -- Dynamic region layout -------------------------------------------
        # Size each region from THIS configuration instead of the fixed gaps
        # the module-level *_BASE_ADDR constants encode. The static kernel gap
        # (0x110000..0x120000 = 64 KiB) silently overflowed into the mask
        # region whenever out_channels * num_passes * 128 exceeded it -- e.g.
        # in_channels=512, out_channels=160 needs 81,920 B and corrupted
        # output channel 128 onward with no error raised. Region sizes are in
        # *narrow-mode* bytes (1 B/element); setup() rescales to the active
        # mode's row size, which preserves the row NUMBERS the CRs carry.
        input_rows = self.row_groups * in_channels
        kernel_rows = out_channels * num_passes
        output_rows = self.row_groups * out_channels
        self._regions = allocate_regions([
            ("input", input_rows * CHUNK_BYTES),
            ("kernel", kernel_rows * CHUNK_BYTES),
            ("mask", CHUNK_BYTES),
            ("output", output_rows * CHUNK_BYTES),
        ])
        self.input_base_row = self._regions["input"] // CHUNK_BYTES
        self.kernel_base_row = self._regions["kernel"] // CHUNK_BYTES
        self.mask_base_row = self._regions["mask"] // CHUNK_BYTES
        self.output_base_row = self._regions["output"] // CHUNK_BYTES
        self.output_base_addr = self._regions["output"]

    def _pack_kernel(self, raw_kernel: bytes) -> bytes:
        """Pack kernel with oc_per_reg=1 layout, zero-padded.

        Raw layout: raw_kernel[oc * in_channels + ic]
        Packed layout (element-identical in both modes; only the byte scale
        differs -- 1 B/element narrow, 4 B/element wide-vector debug, since
        MULT.VE indexes ra by lane in wide mode and by byte in narrow):
          [OC 0, pass 0: 128 elements]
          [OC 0, pass 1: 128 elements]
          ...
          [OC 0, pass P-1: tail padded to 128]
          [OC 1, pass 0: ...]
          ...

        Total = out_channels * num_passes * 128 elements.
        """
        P = self.num_passes
        in_ch = self.in_channels
        out_ch = self.out_channels
        element_width = self._element_width

        def put(packed: bytearray, elem_idx: int, value: int) -> None:
            if element_width == 1:
                packed[elem_idx] = value & 0xFF
            else:
                struct.pack_into("<i", packed, elem_idx * 4, _as_signed_byte(value))

        # Pad out_channels up to even (we pair r0+r1). out_ch % 4 == 0 → already even.
        packed = bytearray(out_ch * P * 128 * element_width)
        for oc in range(out_ch):
            for p in range(P):
                pass_start_ic = p * 128
                # ICs in this pass: 128 (full) or tail_size (last)
                ics_in_pass = 128 if p < P - 1 else self.tail_size
                dst_base = (oc * P + p) * 128
                src_base = oc * in_ch + pass_start_ic
                for i in range(ics_in_pass):
                    put(packed, dst_base + i, raw_kernel[src_base + i])
                # elements [ics_in_pass..128) stay zero (padding)
        return bytes(packed)

    def setup(self, state: "IpuState") -> None:
        # Master ISA: dtype is a state attribute, not a CR register.
        state.dtype = self.dtype

        # Element width of the active mode: 1 B narrow, 4 B wide-vector debug.
        # Row *numbers* handed to CRs are mode-independent, but the host-side
        # byte pokes below must land at the same rows, so they scale by it.
        self._element_width = 4 if state.wide_vector_debug else 1
        row_bytes = CHUNK_BYTES * self._element_width

        # input_path holds the TRUE (unpadded) [in_channels, height, width]
        # tensor -- pad + chunk-pack to the on-device layout here, internal to
        # this method (see the class docstring's file-layout contract).
        input_raw = np.frombuffer(self.input_path.read_bytes(), dtype=np.int8)
        input_chw = input_raw.reshape(self.in_channels, self.height, self.width)
        padded = np.zeros((self.in_channels, self.rows, self.cols), dtype=np.int8)
        padded[:, :self.height, :self.width] = input_chw
        input_data = pack_input_chunked(padded, self.cols)
        state.xmem.write_address(self.input_base_row * row_bytes, input_data)

        kernel_raw = self.kernel_path.read_bytes()
        kernel_packed = self._pack_kernel(kernel_raw)
        state.xmem.write_address(self.kernel_base_row * row_bytes, kernel_packed)

        # Mask polarity (master, 2026-06-14): bit 1 = KEEP lane, bit 0 = ZERO.
        # This app never masks, so slot 0 must be all-ones (keep every lane).
        # The mask blob does NOT widen -- it is 1 bit per lane in both modes --
        # only its row address scales.
        state.xmem.write_address(self.mask_base_row * row_bytes, b"\xff" * 128)

        # Master ISA: CR0 = read-only 0, CR1 = read-only 1 (cannot be overwritten).
        # INPUT_BASE_ROW is 0, so CR0 serves as both the zero constant and the
        # input/cyclic-load base.  The kernel base (nonzero) is relocated to CR14
        # (whose old role, the constant 1 pass decrement, now uses CR1 directly).
        # All of these are XMEM *row* numbers now, not byte addresses -- the
        # emulator scales them by the active mode's row size.
        state.regfile.set_cr(2, self.mask_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(14, self.kernel_base_row)

        # Parameter CR registers (see DESIGN.md). cr8/cr13 are XMEM-space
        # (rows); cr4/cr5/cr6/cr7/cr9/cr10/cr11 are plain counters/lane
        # bounds, unaffected by row addressing.
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
        # CR15.valid_elements = 128 to quantize the full 128-lane output chunk.
        # (Mults use mask_offset 0 / no masking, so partition is irrelevant.)
        state.set_cr_dstructure(valid_elements=128)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_rows = self.row_groups * self.out_channels
            dump_outputs(
                state, self.output_path,
                self.output_base_addr, OUTPUT_ROW_BYTES, total_rows,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain: kernel_size==1
# (pointwise has no 3x3 neighbourhood), groups==1 (a 1x1 depthwise conv has no
# matching app), stride==1, padding==0 (nothing to pad for -- no neighbourhood).
# See _spec_support.ConvQuery for the full query shape shared across this
# package's kernels.


def _supports(**params):
    q = conv_query(**params)
    if bad := positive_dims(q):
        return no(bad)
    if q.kernel_size != 1:
        return no(f"handles only kernel_size=1 (pointwise); got {q.kernel_size}")
    if q.dilation != 1:
        return no(f"handles only dilation=1; got {q.dilation}")
    if q.padding != 0:
        return no(f"handles only padding=0 (no neighbourhood to pad for); got {q.padding}")
    if q.stride != 1:
        return no(f"handles only stride=1; got {q.stride}")
    if q.groups != 1:
        return no(f"handles only groups=1 (a 1x1 depthwise conv has no matching app); got {q.groups}")
    if q.in_channels % 8 != 0:
        return no(f"in_channels ({q.in_channels}) must be a multiple of 8")
    if q.out_channels % 4 != 0:
        return no(f"out_channels ({q.out_channels}) must be a multiple of 4")
    if q.apply_relu:
        return no("apply_relu=True has no matching app here; see pointwise_conv_unified_bn_activation")
    if bad := bias_requires_relu(q):
        return no(bad)
    if q.has_bias:
        return no("bias is not supported by this kernel; see pointwise_conv_unified_bn_activation")
    return yes()


def _build(**params):
    q = conv_query(**params)
    padded_rows, padded_cols = pointwise_pad_shape(q.height, q.width)
    return {
        "rows": padded_rows, "cols": padded_cols,
        "in_channels": q.in_channels, "out_channels": q.out_channels,
    }


def _explain(**params):
    q = conv_query(**params)
    padded_rows, padded_cols = pointwise_pad_shape(q.height, q.width)
    return (
        f"kernel_size=1, groups=1, stride=1, padding=0: the unified pointwise "
        f"kernel. {q.height}x{q.width} pads to {padded_rows}x{padded_cols} "
        "(no mask care needed -- a 1x1 conv has no spatial neighbourhood for a "
        "padded lane to leak into)."
    )


def _caveats(**params):
    q = conv_query(**params)
    padded_rows, padded_cols = pointwise_pad_shape(q.height, q.width)
    if (padded_rows, padded_cols) == (q.height, q.width):
        return ()
    real = q.height * q.width
    padded = padded_rows * padded_cols
    return (
        f"{q.height}x{q.width} pads to {padded_rows}x{padded_cols}, so "
        f"{padded - real} of every {padded} spatial positions idle "
        f"({real / padded:.0%} utilisation).",
    )


SPEC = KernelSpec(
    name="pointwise_conv_unified",
    op="conv2d",
    variant="pointwise",
    app_class=PointwiseConvUnifiedApp,
    asm="pointwise_conv_unified.asm",
    requires=REQUIRES,
    tags=("int8",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: conv_query(**params).bundle,
    # No bias/activation support: strictly cheaper than the BN twin whenever
    # both would otherwise tie (they never overlap today since has_bias/
    # apply_relu split them, but cost still expresses the specialisation).
    cost=lambda **params: 0.0,
)
