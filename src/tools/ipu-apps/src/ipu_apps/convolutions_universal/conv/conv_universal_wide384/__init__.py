"""Wide (W >= 384) standard 3x3 convolution harness, stride 1, FP32.

Handles spatial widths that don't fit the usual <=256-wide chunk-interleaved
layout other conv apps assume. Per the user's explicit addressing spec: one
spatial row of ONE channel occupies ``W // 128`` (``cpr``, "chunks per row")
consecutive XMEM rows, channel-interleaved per spatial row --

    [ch0 row0: cpr rows][ch1 row0: cpr rows]...[ch(C-1) row0: cpr rows]
    [ch0 row1: cpr rows][ch1 row1: cpr rows]...

This is deliberately UNOPTIMIZED (correctness first, per the design brief):
no rotating-slot pipelining. For each (row, column-chunk, filter, channel,
kr) the asm reloads a fresh 3-slot R_CYCLIC strip (the neighbouring
column-chunk to the left, this column-chunk, the neighbouring column-chunk to
the right) and reads the 3 kc taps as plain (non-slot-aligned) 128-element
windows into that strip -- see the .asm module docstring for the full
technique, which generalizes conv_first_layer's 2-slot 256-wide strip trick
to an arbitrary column-chunk boundary (needed because cpr>=3 rows' worth of
3 kr-neighbour rows would overflow the 512-element R_CYCLIC ring if a whole
spatial row were kept resident at once, unlike conv_first_layer's cpr=2 case).

``cpr`` and ``rows`` are Jinja constants baked in at ASSEMBLE time (not
runtime CRs): the row dimension is split into 3 sections mirroring
conv_universal's g0/main/gN split -- the top border row and bottom border row
are each unrolled once (their mask choices are compile-time immediates), but
the interior rows (1..rows-2, which all share the SAME no-vertical-border
mask choices) are emitted as a single GENUINE RUNTIME loop, not unrolled per
row -- unrolling every row would blow past the 1024-instruction program
memory for anything but tiny `rows`. The column-chunk loop (``cpr`` sections,
typically 3-5) IS fully unrolled, since border masking there varies by cc and
cpr is always small. This app therefore renders and assembles its OWN binary
at run() time (bypassing the base class's raw-binary-in / no-Jinja-context
path other apps use), landing the freshly assembled bytes in a tempdir -- no
.bin build artifacts are written into the source tree.

``input_path``/``output_path`` hold the TRUE ``[channels, rows, width]``
float32 tensor -- see ``pointwise_conv_unified``'s class docstring for the
exact file-layout contract this mirrors; the channel-interleaved-per-row
on-device layout is strictly internal.

Usage::

    from ipu_apps.convolutions_universal.conv.conv_universal_wide384 import (
        ConvUniversalWide384App,
    )

    app = ConvUniversalWide384App(
        input_path="input.bin",   # raw [in_channels, rows, width] float32
        kernel=weights,           # np.ndarray [out_ch, in_ch, 3, 3] float32
        output_path="output.bin",
        width=384, rows=16, in_channels=3, out_channels=4,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import jinja2
import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import CHUNK_ELEMENTS, allocate_regions
from ipu_apps.convolutions_universal._spec_support import REQUIRES, conv_query, positive_dims
from ipu_apps.kernel_registry import KernelSpec, no, yes

if TYPE_CHECKING:
    pass

ASM_TEMPLATE_PATH = Path(__file__).resolve().parent / "conv_universal_wide384.asm"

ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode

FPB = 128 // 9  # 14: channels per 128-element dense kernel block

# Mask blob: 4 slots used.
#   0 = KEEP all       -> fully in-bounds taps
#   1 = ZERO all       -> vertical border (kr out of [0, rows)): the whole
#                          neighbour ROW is out of bounds, so every lane of
#                          the tap is invalid.
#   2 = ZERO lane 0     -> TRUE left image edge (cc==0, kc=-1): only output
#                          column 0 (lane 0 of this tap) has no left
#                          neighbour: every other lane is a legitimate
#                          in-chunk read via the rc_idx=127 straddle (see the
#                          .asm module docstring's per-tap 3-slot strip note).
#   3 = ZERO lane 127   -> TRUE right image edge (cc==cpr-1, kc=+1): mirror
#                          of slot 2, only the last lane is invalid.
MASK_SLOT_KEEP = 0
MASK_SLOT_ZERO = 1
MASK_SLOT_ZERO_LANE0 = 2
MASK_SLOT_ZERO_LANE127 = 3


def build_wide384_mask_blob() -> bytes:
    """Build the 128-byte R_MASK blob (see the 4-slot layout above)."""
    mask = bytearray(128)
    for byte_idx in range(16):
        mask[MASK_SLOT_KEEP * 16 + byte_idx] = 0xFF
        mask[MASK_SLOT_ZERO * 16 + byte_idx] = 0x00
        mask[MASK_SLOT_ZERO_LANE0 * 16 + byte_idx] = 0xFF
        mask[MASK_SLOT_ZERO_LANE127 * 16 + byte_idx] = 0xFF
    # slot 2: clear bit 0 (lane 0)
    mask[MASK_SLOT_ZERO_LANE0 * 16 + 0] &= ~(1 << 0)
    # slot 3: clear bit 127 (lane 127) -> byte 15, bit 7
    mask[MASK_SLOT_ZERO_LANE127 * 16 + 15] &= ~(1 << 7)
    return bytes(mask)


def _pack_conv_weights_dense_fp32(weights: np.ndarray) -> bytes:
    """Pack ``[out_ch, in_ch, 3, 3]`` float32 weights into dense 128-element blocks.

    One 128-element block holds ``FPB = 14`` input-channel slots of one
    output filter (9 taps each, 126 of 128 elements used). Blocks for filter
    0 come first (``ceil(in_ch / FPB)`` of them), then filter 1, and so on.
    The last block of each filter zero-pads any unused slots.
    """
    out_ch, in_ch, kh, kw = weights.shape
    if kh != 3 or kw != 3:
        raise ValueError(f"weights trailing dims must be (3, 3), got ({kh}, {kw})")

    blocks_per_filter = math.ceil(in_ch / FPB)
    total_elements = out_ch * blocks_per_filter * 128
    packed = np.zeros(total_elements, dtype=np.float32)
    taps = weights.reshape(out_ch, in_ch, 9)

    for f in range(out_ch):
        for b in range(blocks_per_filter):
            block_base = (f * blocks_per_filter + b) * 128
            for s in range(FPB):
                ic = b * FPB + s
                if ic >= in_ch:
                    break
                dst = block_base + s * 9
                packed[dst:dst + 9] = taps[f, ic]
    return packed.tobytes()


class ConvUniversalWide384App(IpuApp):
    """Wide (W>=384) standard 3x3 convolution, stride 1, FP32.

    Args:
        input_path:   Path to the input image binary, raw ``[in_channels,
                      rows, width]`` float32.
        kernel:       Numpy weights of shape ``[out_ch, in_ch, 3, 3]`` float32.
        kernel_path:  Alternative: path to a raw ``[out_ch, in_ch, 9]``
                      contiguous float32 file.
        output_path:  Optional path to write the output (raw ``[out_channels,
                      rows, width]`` float32).
        width:        Spatial width; multiple of 128, >= 384.
        rows:         Spatial height (>= 1).
        in_channels:  Number of input channels (>= 1).
        out_channels: Number of output channels (>= 1, even per the task
                      brief, though the asm itself does not require it).
    """

    def __init__(
        self,
        *,
        input_path: str | Path,
        width: int,
        rows: int,
        in_channels: int,
        out_channels: int,
        kernel: Optional[np.ndarray] = None,
        kernel_path: Optional[str | Path] = None,
        **kwargs,
    ) -> None:
        kwargs.pop("inst_path", None)  # unused; this app assembles its own binary
        super().__init__(inst_path=ASM_TEMPLATE_PATH, **kwargs)
        self.input_path = Path(input_path)

        if width % 128 != 0 or width < 384:
            raise ValueError(f"width must be a multiple of 128 and >= 384, got {width}")
        if rows < 1:
            raise ValueError(f"rows must be >= 1, got {rows}")
        if in_channels < 1:
            raise ValueError(f"in_channels ({in_channels}) must be >= 1")
        if out_channels < 1:
            raise ValueError(f"out_channels ({out_channels}) must be >= 1")
        if out_channels % 2 != 0:
            raise ValueError(f"out_channels must be even, got {out_channels}")

        if kernel is not None and kernel_path is not None:
            raise ValueError("Provide exactly one of kernel= or kernel_path=")
        if kernel is None and kernel_path is None:
            raise ValueError("Provide one of kernel= or kernel_path=")
        self._kernel_array = kernel
        self.kernel_path = Path(kernel_path) if kernel_path is not None else None

        self.width = width
        self.rows = rows
        self.cpr = width // 128
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks_per_filter = math.ceil(in_channels / FPB)
        # Each dense FPB=14 kernel block is exactly 128 elements = 1 XMEM row,
        # so the per-block reload advance (cr9, added to lr9 in the asm's
        # `_reload` path) is always 1 row -- NOT blocks_per_filter (that would
        # skip whole filters' worth of blocks on every reload once
        # in_channels > FPB). total_kernel_rows (the filter-loop bound,
        # compared against the persistent per-filter kernel base lr8) is the
        # one quantity that actually needs blocks_per_filter.
        self.kernel_row_stride = 1
        self.total_kernel_rows = out_channels * self.blocks_per_filter

        # -- Dynamic region layout -------------------------------------------
        # See conv_universal's identical comment for the full rationale: the
        # fixed *_BASE_ADDR gaps silently overflow at realistic channel
        # counts (kernel size scales with out_channels*in_channels). No
        # guard-band shift is needed here (unlike conv_universal): this app
        # writes input data straight to its region's own base.
        self._regions = allocate_regions([
            ("input", in_channels * rows * self.cpr * CHUNK_ELEMENTS),
            ("kernel", self.total_kernel_rows * CHUNK_ELEMENTS),
            ("mask", CHUNK_ELEMENTS),
            ("output", rows * out_channels * self.cpr * CHUNK_ELEMENTS),
        ])
        self.input_base_row = self._regions["input"] // CHUNK_ELEMENTS
        self.input_base_addr = self._regions["input"] * 4
        self.kernel_base_row = self._regions["kernel"] // CHUNK_ELEMENTS
        self.kernel_base_addr = self._regions["kernel"] * 4
        self.mask_base_row = self._regions["mask"] // CHUNK_ELEMENTS
        self.mask_base_addr = self._regions["mask"] * 4
        self.output_base_row = self._regions["output"] // CHUNK_ELEMENTS
        self.output_base_addr = self._regions["output"] * 4

    def _pack_kernel(self) -> bytes:
        if self._kernel_array is not None:
            weights = np.asarray(self._kernel_array, dtype=np.float32)
            if weights.shape != (self.out_channels, self.in_channels, 3, 3):
                raise ValueError(
                    f"kernel must be [{self.out_channels}, {self.in_channels}, 3, 3], "
                    f"got {weights.shape}"
                )
        else:
            raw = self.kernel_path.read_bytes()
            expected = self.out_channels * self.in_channels * 9 * 4
            if len(raw) != expected:
                raise ValueError(
                    f"kernel_path file has {len(raw)} bytes, expected {expected} "
                    "(out_ch * in_ch * 9 * 4 B float32)"
                )
            weights = (
                np.frombuffer(raw, dtype=np.float32)
                .reshape(self.out_channels, self.in_channels, 3, 3)
            )
        return _pack_conv_weights_dense_fp32(weights)

    def _render_asm(self) -> str:
        text = ASM_TEMPLATE_PATH.read_text()
        template = jinja2.Template(text)
        return template.render(cpr=self.cpr, rows=self.rows)

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
        # input_path holds the TRUE [in_channels, rows, width] float32
        # tensor -- repack to the channel-interleaved-per-row on-device layout.
        raw = np.frombuffer(self.input_path.read_bytes(), dtype=np.float32)
        expected = self.in_channels * self.rows * self.width
        if raw.size != expected:
            raise ValueError(
                f"input has {raw.size} elements, expected {expected} "
                "(in_channels * rows * width)"
            )
        input_chw = raw.reshape(self.in_channels, self.rows, self.width)
        # row(r, ic, cc) = (r * in_channels + ic) * cpr + cc, i.e. transpose
        # to [rows, in_channels, width] then flatten (width already splits
        # into cpr*128 contiguous elements = cpr rows).
        input_data = np.ascontiguousarray(input_chw.transpose(1, 0, 2))
        state.xmem.write_address(
            self.input_base_row * ROW_BYTES, input_data.astype(np.float32).tobytes()
        )

        state.xmem.write_address(self.kernel_base_row * ROW_BYTES, self._pack_kernel())
        # The mask blob does NOT widen (1 bit/lane in both modes) -- only its
        # row address scales.
        state.xmem.write_address(self.mask_base_row * ROW_BYTES, build_wide384_mask_blob())

        state.set_cr_dstructure(valid_elements=128)  # Partition.P0 default: whole-tap masking

        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.kernel_base_row)
        state.regfile.set_cr(4, self.output_base_row)
        state.regfile.set_cr(5, self.mask_base_row)
        state.regfile.set_cr(6, self.in_channels)
        # cr7 = interior-row runtime-loop limit: out_row_base (RELATIVE to
        # cr4, like lr7 -- see the asm init section) of the LAST row (row
        # `rows-1`). The interior loop (rows 1..rows-2) compares lr7
        # (out_row_base, advanced by cr10 each row) against this and stops
        # BEFORE processing the last row, which is handled by its own
        # (bottom-border) unrolled section. Only meaningful when rows > 2;
        # harmless to always set.
        state.regfile.set_cr(7, (self.rows - 1) * self.out_channels * self.cpr)
        state.regfile.set_cr(8, FPB)
        state.regfile.set_cr(9, self.kernel_row_stride)
        state.regfile.set_cr(10, self.out_channels * self.cpr)
        state.regfile.set_cr(11, self.in_channels * self.cpr)
        state.regfile.set_cr(12, self.cpr)
        state.regfile.set_cr(13, self.total_kernel_rows)
        state.regfile.set_cr(14, self.blocks_per_filter)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.rows * self.out_channels * self.cpr
            total_elements = total_outputs * CHUNK_ELEMENTS
            raw = state.xmem.read_address(
                self.output_base_row * ROW_BYTES, total_elements * 4
            )
            out = np.frombuffer(raw, dtype=np.float32).reshape(
                self.rows, self.out_channels, self.width,
            )
            out_chw = np.ascontiguousarray(out.transpose(1, 0, 2))
            Path(self.output_path).write_bytes(out_chw.astype(np.float32).tobytes())

    def run(self, *, max_cycles: int = 5_000_000, **kwargs) -> tuple["IpuState", int]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="conv_universal_wide384_"))
        bin_path = tmp_dir / "conv_universal_wide384.bin"
        assemble_to_bin_file(self._render_asm(), str(bin_path))
        self.inst_path = bin_path
        kwargs.setdefault("state", self.make_state())
        return super().run(max_cycles=max_cycles, **kwargs)


# -- registry declaration ---------------------------------------------------


def _supports(**params):
    q = conv_query(**params)
    if bad := positive_dims(q):
        return no(bad)
    if q.kernel_size != 3:
        return no(f"handles only kernel_size=3; got {q.kernel_size}")
    if q.dilation != 1:
        return no(f"handles only dilation=1; got {q.dilation}")
    if q.padding != 1:
        return no(f"handles only padding=1; got {q.padding}")
    if q.stride != 1:
        return no(f"handles only stride=1; got {q.stride}")
    if q.groups != 1:
        return no(f"handles only groups=1 (plain conv); got {q.groups}")
    if q.apply_relu:
        return no("apply_relu=True is not supported by this kernel")
    if q.has_bias:
        return no("bias is not supported by this kernel")
    if q.width % 128 != 0 or q.width < 384:
        return no(
            f"handles only width a multiple of 128 and >= 384 (see "
            f"conv_universal for narrower widths); got {q.width}"
        )
    if q.out_channels % 2 != 0:
        return no(f"out_channels must be even; got {q.out_channels}")
    return yes()


def _build(**params):
    q = conv_query(**params)
    return {
        "width": q.width, "rows": q.height,
        "in_channels": q.in_channels, "out_channels": q.out_channels,
    }


def _explain(**params):
    return (
        "kernel_size=3, groups=1, stride=1, padding=1, width%128==0 and "
        ">=384: the wide-image 3x3 conv kernel (FP32), unoptimized "
        "(no rotating-slot pipelining) but handles arbitrarily wide images."
    )


def _caveats(**params):
    return (
        "FP32 wide-vector debug mode only (wide_vector_debug=True). This "
        "kernel has no INT8/quantized variant.",
        "Deliberately unoptimized: no rotating-slot pipelining, reloads a "
        "fresh 3-slot strip per tap.",
    )


SPEC = KernelSpec(
    name="conv_universal_wide384",
    op="conv2d",
    variant="wide384",
    app_class=ConvUniversalWide384App,
    asm="conv_universal_wide384.asm",
    requires=REQUIRES,
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: conv_query(**params).bundle,
    cost=lambda **params: 1.0,  # unoptimized fallback: prefer conv_universal when it applies
)
