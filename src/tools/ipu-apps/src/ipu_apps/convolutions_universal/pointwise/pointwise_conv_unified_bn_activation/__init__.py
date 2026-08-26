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
  - spatial: cols divides 128 (power of 2, 1..128); rows >= 1;
    rows*cols % 128 == 0 (a whole number of 128-byte chunks)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_math import DType

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    parse_dtype,
    dump_outputs,
    allocate_regions,
)
from ipu_apps.convolutions_universal._spec_support import (
    REQUIRES,
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
# actually loads/stores through -- including the new BIAS_BASE_ROW, which
# feeds cr10 exactly like KERNEL_BASE_ROW feeds cr14 (same LDR_MULT_REG
# pattern, same row-number treatment). r_cyclic ELEMENT addressing is
# untouched -- see the .asm header for the full recipe note.

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x110000
MASK_BASE_ADDR = 0x120000
BIAS_BASE_ADDR = 0x130000
# Output can span up to row_groups * out_ch * 128 bytes; for the largest config
# (64x64, oc=64) that is 0x40000, so OUTPUT must sit clear of the bias region.
OUTPUT_BASE_ADDR = 0x140000

OUTPUT_ROW_BYTES = 128
CHUNK_BYTES = 128

INPUT_BASE_ROW = INPUT_BASE_ADDR // CHUNK_BYTES
KERNEL_BASE_ROW = KERNEL_BASE_ADDR // CHUNK_BYTES
MASK_BASE_ROW = MASK_BASE_ADDR // CHUNK_BYTES
BIAS_BASE_ROW = BIAS_BASE_ADDR // CHUNK_BYTES
OUTPUT_BASE_ROW = OUTPUT_BASE_ADDR // CHUNK_BYTES


def _as_signed_byte(value: int) -> int:
    """Reinterpret a wire byte as the signed INT8 it encodes."""
    v = value & 0xFF
    return v - 256 if v > 127 else v


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
        rows:         Spatial height (>= 1; rows*cols must be a multiple of 128).
        cols:         Spatial width (must divide 128: 1, 2, 4, ..., 128).
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

        # Spatial constraints, as the hardware actually imposes them.
        #
        # The .asm never sees `rows` or `cols` -- the only shape parameter
        # reaching it is cr5 = row_groups = rows*cols/128 (see the CR map in
        # the .asm header).  `rows_per_chunk` is used purely host-side by the
        # packing helpers.  So the real requirements are:
        #
        #   * `cols` must divide 128, so that whole spatial rows tile a
        #     128-byte chunk without straddling its edge (the packing loop
        #     writes packed[dst + r*cols : dst + r*cols + cols]).  The
        #     divisors of 128 are exactly the powers of two <= 128.
        #   * rows*cols must be a whole number of 128-byte chunks.
        #
        # The previous check also demanded rows be a power of two in
        # [16..128].  That had no hardware basis and rejected legitimate
        # shapes such as 8x16 (one full chunk, row_groups=1), which is what a
        # padded 8x8 pointwise layer becomes.  This is a pure widening: every
        # shape accepted before is still accepted.
        valid_cols = {1, 2, 4, 8, 16, 32, 64, 128}
        if cols not in valid_cols:
            raise ValueError(
                f"cols must divide 128 (one of {sorted(valid_cols)}), got {cols}"
            )
        if rows < 1:
            raise ValueError(f"rows must be >= 1, got {rows}")
        if (rows * cols) % 128 != 0:
            raise ValueError(
                f"rows*cols ({rows}*{cols} = {rows * cols}) must be a multiple "
                "of 128 (a whole number of 128-byte chunks)"
            )

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
        # See the sibling pointwise_conv_unified for the full rationale: the
        # fixed 64 KiB kernel gap silently overflowed into the next region
        # once out_channels * num_passes * 128 exceeded it. The bias region
        # mirrors the kernel's block grid exactly (see _pack_bias), so it
        # overflows on precisely the same configurations and is sized the
        # same way here.
        input_rows = self.row_groups * in_channels
        kernel_rows = out_channels * num_passes
        output_rows = self.row_groups * out_channels
        self._regions = allocate_regions([
            ("input", input_rows * CHUNK_BYTES),
            ("kernel", kernel_rows * CHUNK_BYTES),
            ("mask", CHUNK_BYTES),
            ("bias", kernel_rows * CHUNK_BYTES),
            ("output", output_rows * CHUNK_BYTES),
        ])
        self.input_base_row = self._regions["input"] // CHUNK_BYTES
        self.kernel_base_row = self._regions["kernel"] // CHUNK_BYTES
        self.mask_base_row = self._regions["mask"] // CHUNK_BYTES
        self.bias_base_row = self._regions["bias"] // CHUNK_BYTES
        self.output_base_row = self._regions["output"] // CHUNK_BYTES
        self.output_base_addr = self._regions["output"]

    def _pack_kernel(self, raw_kernel: bytes) -> bytes:
        """Pack kernel with oc_per_reg=1 layout, zero-padded.

        Raw layout: raw_kernel[oc * in_channels + ic]
        Packed layout (element-identical in both modes; only the byte scale
        differs -- 1 B/element narrow, 4 B/element wide-vector debug):
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

    def _pack_bias(self) -> bytes:
        """Pack per-OC INT8 bias into a kernel-mirroring region.

        Region shape = out_ch × num_passes × 128 (identical block grid to the
        packed kernel; element-identical across modes -- see _pack_kernel).
        The OC's bias goes in element 0 of its **pass-0** block; every other
        element is zero. The asm indexes this with ``lr12`` (the kernel row
        offset, which sits at the OC's pass-0 block at OC entry) via cr10, so
        no separate pointer is needed.
        """
        P = self.num_passes
        out_ch = self.out_channels
        element_width = self._element_width
        bias_bytes = self._bias_array.astype(np.int8).view(np.uint8)

        packed = bytearray(out_ch * P * 128 * element_width)
        for oc in range(out_ch):
            # element 0 of OC's pass-0 block (== same offset as kernel pass-0 block)
            if element_width == 1:
                packed[(oc * P) * 128] = int(bias_bytes[oc])
            else:
                struct.pack_into(
                    "<i", packed, (oc * P) * 128 * 4,
                    _as_signed_byte(int(bias_bytes[oc])),
                )
        return bytes(packed)

    def setup(self, state: "IpuState") -> None:
        # Master ISA: dtype is a state attribute, not a CR register.
        state.dtype = self.dtype

        # Element width of the active mode: 1 B narrow, 4 B wide-vector debug.
        # Row *numbers* handed to CRs are mode-independent, but the host-side
        # byte pokes below must land at the same rows, so they scale by it.
        self._element_width = 4 if state.wide_vector_debug else 1
        row_bytes = CHUNK_BYTES * self._element_width

        input_data = self.input_path.read_bytes()
        state.xmem.write_address(self.input_base_row * row_bytes, input_data)

        kernel_raw = self.kernel_path.read_bytes()
        kernel_packed = self._pack_kernel(kernel_raw)
        state.xmem.write_address(self.kernel_base_row * row_bytes, kernel_packed)

        # Folded-bias region (mirrors the kernel layout — see _pack_bias).
        state.xmem.write_address(self.bias_base_row * row_bytes, self._pack_bias())

        # Mask polarity (master, 2026-06-14): bit 1 = KEEP lane, bit 0 = ZERO.
        # This app never masks, so slot 0 must be all-ones (keep every lane).
        # The mask blob does NOT widen -- only its row address scales.
        state.xmem.write_address(self.mask_base_row * row_bytes, b"\xff" * 128)

        # Master ISA: CR0 = read-only 0, CR1 = read-only 1 (cannot be overwritten).
        # INPUT_BASE_ROW is 0, so CR0 serves as both the zero constant and the
        # input/cyclic-load base.  The kernel base (nonzero) is relocated to CR14
        # (whose old role, the constant 1 pass decrement, now uses CR1 directly).
        # All of these are XMEM *row* numbers now, not byte addresses.
        state.regfile.set_cr(2, self.mask_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(14, self.kernel_base_row)

        # Parameter CR registers (see DESIGN.md)
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

        # Master ISA: ACTIVATE.QUANTIZE reads its active-lane count from the named
        # dstructure CR's valid_elements field. The asm names CR15, so set
        # CR15.valid_elements = 128 to activate+quantize the full 128-lane chunk.
        # (Mults use mask_offset 0 / no masking, so partition is irrelevant; cr10
        # holds the bias base, cr15 is otherwise free for the dstructure config.)
        state.set_cr_dstructure(valid_elements=128)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_rows = self.row_groups * self.out_channels
            dump_outputs(
                state, self.output_path,
                self.output_base_addr, OUTPUT_ROW_BYTES, total_rows,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. Same
# domain as pointwise_conv_unified except this twin REQUIRES apply_relu=True
# (it unconditionally applies ReLU) -- see _spec_support.bias_requires_relu.
#
# `bias` (the actual tensor, not just has_bias:bool) is not part of the
# registry query -- like every other kernel here, the registry carries only
# shapes/config, never tensor data. A caller that resolves this kernel passes
# its own quantized bias array to the constructor directly (see
# convolutions_universal.layers.run_layer).


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
    if not q.apply_relu:
        return no(
            "this kernel unconditionally applies ReLU; apply_relu=False has "
            "no matching app here (see pointwise_conv_unified for the "
            "no-bias/no-activation path)"
        )
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
        f"kernel_size=1, groups=1, stride=1, padding=0, apply_relu=True: the "
        f"unified pointwise kernel with folded bias + ReLU. {q.height}x"
        f"{q.width} pads to {padded_rows}x{padded_cols} (no mask care needed)."
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
    name="pointwise_conv_unified_bn_activation",
    op="conv2d",
    variant="pointwise_bn_activation",
    app_class=PointwiseConvUnifiedBnActivationApp,
    asm="pointwise_conv_unified_bn_activation.asm",
    requires=REQUIRES,
    tags=("int8",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: conv_query(**params).bundle,
    cost=lambda **params: 0.0,
)
