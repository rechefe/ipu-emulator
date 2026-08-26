"""Universal depthwise 3x3 convolution harness (no bias, no activation, FP32).

Base app for ``depthwise_conv_universal_bn_activation``. Same chunk-interleaved
I/O layout and walking-pointer / rotating-cyclic-slot pipeline, minus:

  * **no folded bias** — ``r_acc`` is seeded by tap 1's own product
    (``acc.add.first``) instead of a bias multiply-broadcast.
  * **no ReLU** — ``ACTIVATE`` runs with ``identity``.

Per-channel budget: **10 cyc/ch** = 9 weight taps + 1 ACTIVATE cycle (tap 1
doubles as the r_acc reset via ``acc.add.first``, replacing the BN twin's
separate bias-seed cycle — the one cycle actually saved). ACTIVATE still
needs its own cycle (reads the cycle-start snapshot of ``r_acc``, same as the
BN twin's placeholder cycle) and now also co-issues the next channel's kr=-1
prefetch load, since ACTIVATE occupies its own slot type and leaves the
LR/XMEM slots free that cycle — replacing the load the BN twin's bias-seed
cycle used to carry. The deferred-store pipeline (store the previous
channel's result while the current channel computes) is preserved. Runs on
the emulator's wide-vector debug datapath (FP32) -- weights and activations
are genuine floats, no INT8 quantization anywhere in this kernel.

Kernel super-block layout (FPB=28, 9-element stride, no bias element):
  Depthwise produces one output PER channel; each channel occupies a **9-element
  slot** (its 9 weight taps only — no bias element to reserve). 28 channels * 9 =
  252 <= 256, so one 256-element super-block (R0 = elements 0..127, R1 = 128..255)
  holds 28 channels. The shared ``mult.ve`` fixed_idx (0..255) addresses both
  halves transparently.

  The asm walks one continuous kernel element index ``lr6`` at +1 per cycle: for
  channel ``s`` taps 1..9 read ``fixed_idx = s*9 .. s*9 + 9``; the next
  channel's tap 1 is the following element, so the 9-cycle/channel body advances
  ``lr6`` by exactly one channel stride.

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
        DepthwiseConvUniversalApp,
    )

    app = DepthwiseConvUniversalApp(
        inst_path="depthwise_conv_universal.bin",
        input_path="input.bin",      # raw [channels, height, width] float32
        kernel=weights,               # np.ndarray [channels, 3, 3] float32
        output_path="output.bin",
        height=64, width=64, channels=256,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_config import Partition

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    CHUNK_ELEMENTS,
    dump_outputs,
    allocate_regions,
    pack_input_chunked,
)
from ipu_apps.convolutions_universal._spec_support import (
    REQUIRES,
    conv_query,
    min_rows_for_chunk_floor,
    next_valid_cols,
    positive_dims,
)
# Reuse the conv_universal_bn_activation mask-blob builder so the depthwise
# apps share one border-mask implementation: a single 128-byte blob (slots
# 0/3/6) where left/right edge columns are applied at runtime via mask_shift
# (CR15 partition).
from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
    build_border_mask_blob,
)
from ipu_apps.kernel_registry import KernelSpec, no, yes

if TYPE_CHECKING:
    pass

# -- Memory layout -----------------------------------------------------------
#
# Row-addressed ISA (mb/195): XMEM offset/base operands on LDR_*/STR_*
# (including LDR_CYCLIC_MULT_REG's offset/base -- only its `index` is
# r_cyclic-element-space) are ROW numbers, not byte addresses. This app runs
# FP32 wide-vector only, so ROW_BYTES is always 512.
#
# r_cyclic ELEMENT addressing is untouched by this migration: lr5 (the
# LDR_CYCLIC_MULT_REG `index`) and lr3/lr4 (MULT.RC.VE `rc_idx` / read-slot
# rotation) index a 512-ELEMENT ring in both modes, so cr12/cr13/cr9/cr14 in
# their r_cyclic role (slot-size 128, slot-step 256/384) are unchanged.

ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode

OUTPUT_CHUNK_BYTES = CHUNK_ELEMENTS * 4  # bytes per output channel per chunk (FP32)

FPB = 28           # channels per 256-element super-block (9 taps each, no bias)
CH_SLOT_ELEMENTS = 9  # per-channel slot: 9 weight taps, no bias element
SUPER_BLOCK_ELEMENTS = 256
SUPER_BLOCK_ROWS = SUPER_BLOCK_ELEMENTS * 4 // ROW_BYTES  # = 2


def _pack_depthwise_kernel(kernel_raw: np.ndarray, channels: int) -> bytes:
    """Pack per-channel 9 weight taps into FPB=28 super-blocks (float32).

    Input:  ``kernel_raw`` = [channels, 9] float32 (channel ch's 9 taps).
    Output: ceil(channels/28) super-blocks of ``256`` elements each.

    Within one super-block, channel ``s`` (0..27) occupies ELEMENTS
    ``[s*9 .. s*9 + 9)``. 28*9 = 252 <= 256.  R0 holds elements 0..127, R1
    holds 128..255; the shared-index ``mult.ve`` (fixed_idx 0..255) spans both
    halves.
    """
    num_blocks = math.ceil(channels / FPB)
    total_elements = num_blocks * SUPER_BLOCK_ELEMENTS
    packed = np.zeros(total_elements, dtype=np.float32)

    for sb in range(num_blocks):
        sb_base = sb * SUPER_BLOCK_ELEMENTS
        for s in range(FPB):
            ch = sb * FPB + s
            if ch >= channels:
                break
            slot = sb_base + s * CH_SLOT_ELEMENTS
            packed[slot:slot + 9] = kernel_raw[ch]
    return packed.tobytes()


class DepthwiseConvUniversalApp(IpuApp):
    """Universal depthwise 3x3 convolution harness (no bias, no activation, FP32).

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor -- see
    ``pointwise_conv_unified``'s class docstring for the exact file-layout
    contract this mirrors.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary, raw ``[channels, height,
                      width]`` float32.
        kernel:       Numpy weights of shape ``[channels, 3, 3]`` float32.
        kernel_path:  Alternative: path to a raw ``[channels, 9]`` contiguous
                      float32 file.
        output_path:  Optional path to write output (raw ``[channels, height,
                      width]`` float32).
        height:       Spatial height (>= 1). Padded internally.
        width:        Spatial width (>= 1). Padded internally.
        channels:     Number of channels (>= 1).
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        channels: int,
        kernel: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

        kernel_path = getattr(self, "kernel_path", None)
        if kernel is not None and kernel_path is not None:
            raise ValueError("Provide exactly one of kernel= or kernel_path=")
        if kernel is None and kernel_path is None:
            raise ValueError("Provide one of kernel= or kernel_path=")
        self._kernel_array = kernel
        self.kernel_path = Path(kernel_path) if kernel_path is not None else None

        if height < 1:
            raise ValueError(f"height must be >= 1, got {height}")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if channels < 1:
            raise ValueError(f"channels ({channels}) must be >= 1")

        self.height = height
        self.width = width
        cols = next_valid_cols(width)
        rows = min_rows_for_chunk_floor(height, cols)
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.num_chunks = (rows * cols) // CHUNK_ELEMENTS
        # group_stride is a row-count (XMEM-space): one "chunk group" of
        # `channels` rows, one row per channel.
        self.group_stride = channels
        self.num_super_blocks = math.ceil(channels / FPB)
        self.total_kernel_rows = self.num_super_blocks * SUPER_BLOCK_ROWS
        self.total_kernel_elements = self.num_super_blocks * SUPER_BLOCK_ELEMENTS

        self._compute_regions()

    def _compute_regions(self) -> None:
        """(Re)computes the dynamic region layout from self.num_chunks/group_stride/
        total_kernel_rows/channels.

        Split out from __init__ so a subclass that corrects self.num_chunks
        AFTER calling super().__init__() (see
        depthwise_conv_stride2_128._Stage1FullWidthApp, which temporarily
        passes cols=64 to bypass this class's cols-in-{16,32,64} check, then
        fixes self.cols/self.num_chunks to the true cols=128 values) can
        re-run this to get region sizes that reflect the TRUE shape, not the
        placeholder one __init__ saw.

        See conv_universal's identical comment for why fixed *_BASE_ADDR
        gaps are replaced: they silently overflow at realistic channel
        counts. Depthwise's kernel scales with channels alone (not
        out_ch*in_ch), so it is harder to hit than conv_universal's, but
        the same guard-band logic applies: the g0 section's kr=-1 prefetch
        computes `lr8 - group_stride` at chunk 0, which must not go
        negative, so the real input data is placed one group_stride
        further into the input region than its base -- the "input"
        region's real size is the headroom PLUS the data, not just the
        data.
        """
        input_region_rows = self.group_stride + self.num_chunks * self.group_stride
        self._regions = allocate_regions([
            ("input", input_region_rows * CHUNK_ELEMENTS),
            ("kernel", self.total_kernel_elements),
            ("mask", CHUNK_ELEMENTS),
            ("output", self.num_chunks * self.channels * CHUNK_ELEMENTS),
        ])
        self.input_base_row = self._regions["input"] // CHUNK_ELEMENTS
        self.kernel_base_row = self._regions["kernel"] // CHUNK_ELEMENTS
        self.mask_base_row = self._regions["mask"] // CHUNK_ELEMENTS
        self.output_base_row = self._regions["output"] // CHUNK_ELEMENTS
        self.output_base_addr = self._regions["output"] * 4  # bytes, FP32
        self.input_data_row = self.input_base_row + self.group_stride

    def _pack_kernel(self) -> bytes:
        if self._kernel_array is not None:
            weights = np.asarray(self._kernel_array, dtype=np.float32)
        else:
            raw = self.kernel_path.read_bytes()
            expected = self.channels * 9 * 4
            if len(raw) != expected:
                raise ValueError(
                    f"kernel_path file has {len(raw)} bytes, "
                    f"expected {expected} (channels * 9 * 4 B float32)"
                )
            weights = (
                np.frombuffer(raw, dtype=np.float32).reshape(self.channels, 3, 3)
            )
        w_flat = weights.reshape(self.channels, 9)
        return _pack_depthwise_kernel(w_flat, self.channels)

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
        # input_path holds the TRUE (unpadded) [channels, height, width]
        # float32 tensor -- pad + chunk-pack to the on-device layout here.
        input_raw = np.frombuffer(self.input_path.read_bytes(), dtype=np.float32)
        input_chw = input_raw.reshape(self.channels, self.height, self.width)
        padded = np.zeros((self.channels, self.rows, self.cols), dtype=np.float32)
        padded[:, :self.height, :self.width] = input_chw
        input_data = pack_input_chunked(padded, self.cols)
        state.xmem.write_address(self.input_data_row * ROW_BYTES, input_data)

        kernel_packed = self._pack_kernel()
        state.xmem.write_address(self.kernel_base_row * ROW_BYTES, kernel_packed)

        # Border mask: a SINGLE blob carrying all 3 slots (0=none, 3=top-row
        # zero, 6=bottom-row zero), loaded once at init.  The g0 section selects
        # slot 3, the gN section selects slot 6 — no mid-program R_MASK reload.
        # Left/right edge columns are applied at runtime by mask_shift (CR15
        # partition below).  No zero region.
        state.xmem.write_address(self.mask_base_row * ROW_BYTES, build_border_mask_blob(self.cols))

        # CR15 dstructure: partition so each partition group is exactly one
        # packed spatial row (group size == cols).  The asm's mask_shift then
        # injects the left/right edge-column zero at each packed-row boundary.
        cols_to_partition = {
            128: Partition.P0,  # 1 group of 128 lanes (one packed row per chunk)
            64: Partition.P2,   # 2 groups of 64 lanes
            32: Partition.P4,   # 4 groups of 32 lanes
            16: Partition.P8,   # 8 groups of 16 lanes
        }
        state.set_cr_dstructure(
            valid_elements=128,
            partition=cols_to_partition[self.cols],
        )

        # CR map (master ISA: CR0 = read-only 0, CR1 = read-only 1, CR15 = dstructure).
        # Relocate the input/kernel bases off CR0/CR1 (mirroring
        # conv_universal_bn_activation), keeping CR0 free as the read-only zero
        # constant used by "SET lr<n>, cr0":
        #   CR10 = INPUT_BASE_ROW (cyclic-load base; the asm's own running
        #   pointer lr8/lr2 already carries the guard-band group_stride, which
        #   is why the DATA itself is written at input_data_row -- cr10 stays
        #   at the un-shifted base or the shift would be applied twice).
        #   CR5 = KERNEL_BASE (row), CR3 = mask blob row (single blob, slots 0/3/6).
        state.regfile.set_cr(10, self.input_base_row)
        state.regfile.set_cr(5, self.kernel_base_row)
        # cr2 is pre-biased by -1 ROW for the deferred store (asm advances lr7
        # BEFORE the XMEM store at tap 2; store writes to lr7_advanced + cr2 =
        # lr7_old + OUTPUT_BASE_ROW).
        state.regfile.set_cr(2, (self.output_base_row - 1) & 0xFFFFFFFF)
        state.regfile.set_cr(3, self.mask_base_row)
        # cr9 = 384: r_cyclic slot-pointer step (+384 mod 512 ELEMENTS) for the
        # running write pointer lr5 -- element-space, unchanged by row addressing.
        state.regfile.set_cr(9, 384)

        # Parameter CR registers
        state.regfile.set_cr(4, self.cols)
        # cr6 = group_stride in ROWS (XMEM-space: feeds LDR_CYCLIC_MULT_REG's
        # offset/base sum via lr2/lr14, and the chunk/loop-limit comparisons).
        state.regfile.set_cr(6, self.group_stride)
        state.regfile.set_cr(7, FPB)               # channel group inner-loop size, in rows
        state.regfile.set_cr(8, self.total_kernel_rows)
        # cr11: chunk-loop limit, biased by the same guard-band group_stride
        # added to cr10/input_data_row above.
        state.regfile.set_cr(
            11, (self.num_chunks - 1) * self.group_stride + self.group_stride,
        )
        # cr12 = 128: r_cyclic slot size (ELEMENT-space; index step for lr5).
        # This is NOT the XMEM chunk stride anymore -- that role is CR1
        # (read-only 1: one XMEM chunk == one row).
        state.regfile.set_cr(12, 128)
        state.regfile.set_cr(13, 256)  # r_cyclic half-slot step, element-space
        state.regfile.set_cr(14, (256 - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.channels
            raw = state.xmem.read_address(self.output_base_addr, total_outputs * ROW_BYTES)
            from ipu_apps.convolutions_universal import unpack_output_chunked
            padded_out = unpack_output_chunked(raw, self.channels, self.rows, self.cols)
            out = padded_out[:, :self.height, :self.width]
            self.output_path.write_bytes(out.astype(np.float32).tobytes())

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------
# groups == in_channels (depthwise); no bias, no ReLU here.


def _supports(**params):
    q = conv_query(**params)
    if bad := positive_dims(q):
        return no(bad)
    if q.kernel_size != 3:
        return no(f"handles only kernel_size=3; got {q.kernel_size}")
    if q.dilation != 1:
        return no(f"handles only dilation=1; got {q.dilation}")
    if q.padding != 1:
        return no(f"handles only padding=1 (\"same\" padding for a 3x3 kernel); got {q.padding}")
    if q.stride != 1:
        return no(f"handles only stride=1; got {q.stride}")
    if not q.is_depthwise:
        return no(f"handles only depthwise (groups==in_channels); got groups={q.groups}, in_channels={q.in_channels}")
    if q.out_channels != q.in_channels:
        return no(f"depthwise requires out_channels==in_channels; got {q.out_channels} vs {q.in_channels}")
    if q.apply_relu:
        return no("apply_relu=True has no matching app here; see depthwise_conv_universal_bn_activation")
    if q.has_bias:
        return no("bias is not supported by this kernel; see depthwise_conv_universal_bn_activation")
    return yes()


def _build(**params):
    q = conv_query(**params)
    return {"height": q.height, "width": q.width, "channels": q.in_channels}


def _explain(**params):
    q = conv_query(**params)
    cols = next_valid_cols(q.width)
    rows = min_rows_for_chunk_floor(q.height, cols)
    return (
        f"kernel_size=3, groups=in_channels (depthwise), stride=1, padding=1: "
        f"the universal depthwise 3x3 conv kernel (FP32). {q.height}x{q.width} "
        f"pads internally to {rows}x{cols}."
    )


def _caveats(**params):
    q = conv_query(**params)
    cols = next_valid_cols(q.width)
    rows = min_rows_for_chunk_floor(q.height, cols)
    caveats = (
        "FP32 wide-vector debug mode only (wide_vector_debug=True). This "
        "kernel has no INT8/quantized variant.",
    )
    if (rows, cols) == (q.height, q.width):
        return caveats
    real = q.height * q.width
    padded = rows * cols
    return caveats + (
        f"{q.height}x{q.width} pads to {rows}x{cols}, so "
        f"{padded - real} of every {padded} spatial positions idle "
        f"({real / padded:.0%} utilisation).",
    )


SPEC = KernelSpec(
    name="depthwise_conv_universal",
    op="conv2d",
    variant="depthwise",
    app_class=DepthwiseConvUniversalApp,
    asm="depthwise_conv_universal.asm",
    requires=REQUIRES,
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: conv_query(**params).bundle,
    cost=lambda **params: 0.0,
)
