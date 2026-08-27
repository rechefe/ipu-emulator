"""Depthwise 3x3 stride-2 convolution, the degenerate 16x16 shape:
16x16xC -> 8x8xC. FP32 wide-vector mode.

Closes a real coverage gap: at width=16, height=16 (MobileViT-S's deepest MV2
downsample), the general ``depthwise_conv_stride2_narrow`` sibling cannot
represent this shape at all -- its stage 2 always reads FOUR stage-1
row-groups per output chunk (packing (rows_per_chunk/2)*(cols/2) = 32
elements from each of 4 ACC.STRIDE calls into one 128-element chunk), but at
height=16 there are only TWO stage-1 row-groups total (rows_per_chunk=8 at
width=16). See the ``mobilevit_s_registry_coverage`` memory note.

Two-stage design, same philosophy as both stride2 siblings (build on TOP of
the already-verified ``depthwise_conv_universal`` rather than a fresh
from-scratch pipeline):

  Stage 1: run ``depthwise_conv_universal``'s OWN asm, completely UNMODIFIED,
  against the full-resolution 16x16 input (width=16 is natively supported by
  the base app -- no subclass needed, unlike the width=128 sibling). Produces
  a full-width depthwise conv result, chunk-interleaved, 2 row-groups/channel.

  Stage 2 (``decimate_stage2.asm``, new, fixed shape -- no Jinja templating
  needed since nothing varies but ``channels``): one channel's TRUE output is
  exactly 8x8 = 64 elements -- HALF of one 128-element chunk (2 ACC.STRIDE
  calls * 32 elements/call). Rather than pad the other half with garbage (as
  a naive port of the general narrow kernel's 4-row-group grouping would
  need), this kernel packs TWO adjacent channels into one output chunk:
  channel 2p's two row-groups fill r_acc[0:64] (ACC.STRIDE offsets 0,1),
  channel 2p+1's fill r_acc[64:128] (offsets 2,3) -- exactly one full chunk,
  zero padding, zero wasted computation. ``channels`` must therefore be even.

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_16 import (
        DepthwiseConvStride2_16App,
    )

    app = DepthwiseConvStride2_16App(
        input_path="input.bin",     # raw [channels, 16, 16] float32
        kernel=weights,              # np.ndarray [channels, 3, 3] float32
        output_path="output.bin",
        channels=320,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.emulator import run_test
from ipu_emu.ipu_config import Partition

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import CHUNK_ELEMENTS
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
)
from ipu_apps.convolutions_universal._spec_support import REQUIRES, conv_query, positive_dims
from ipu_apps.kernel_registry import KernelSpec, no, yes

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState


STAGE1_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "depthwise_conv_universal" / "depthwise_conv_universal.asm"
)
STAGE2_ASM_PATH = Path(__file__).resolve().parent / "decimate_stage2.asm"

ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode

ROWS = 16
COLS = 16
ROWS_PER_CHUNK = 128 // COLS  # 8
IN_ROW_GROUPS = ROWS // ROWS_PER_CHUNK  # 2
OUT_ROWS = ROWS // 2  # 8
OUT_COLS = COLS // 2  # 8


class DepthwiseConvStride2_16App(IpuApp):
    """Depthwise 3x3 stride-2 conv, fixed 16x16 input, FP32 (two-stage).

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor -- see
    ``pointwise_conv_unified``'s class docstring for the exact file-layout
    contract this mirrors.

    Args:
        input_path:   Path to input image binary, raw ``[channels, 16, 16]``
                      float32.
        kernel:       Numpy weights of shape ``[channels, 3, 3]`` float32.
        kernel_path:  Alternative: path to a raw ``[channels, 9]`` contiguous
                      float32 file.
        output_path:  Optional path to write output (raw ``[channels, 8,
                      8]`` float32).
        channels:     Number of channels (>= 2, must be even -- two channels
                      pack into one output chunk).
    """

    SELF_ASSEMBLES = True

    def __init__(
        self,
        *,
        channels: int,
        kernel: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        kwargs.pop("inst_path", None)  # unused; stages assemble their own
        super().__init__(inst_path=STAGE1_ASM_PATH, **kwargs)
        self.input_path = Path(self.input_path)

        kernel_path = getattr(self, "kernel_path", None)
        if kernel is not None and kernel_path is not None:
            raise ValueError("Provide exactly one of kernel= or kernel_path=")
        if kernel is None and kernel_path is None:
            raise ValueError("Provide one of kernel= or kernel_path=")
        self.kernel_path = Path(kernel_path) if kernel_path is not None else None

        if channels < 2 or channels % 2 != 0:
            raise ValueError(f"channels must be even and >= 2, got {channels}")

        self.channels = channels
        self.num_channel_pairs = channels // 2

        # Build stage 1 now (paths are fixed up with real files in run(), but
        # __init__ needs no real file to exist -- IpuApp.__init__ only stores
        # Path objects). This is the SAME instance run() actually executes,
        # so its region layout (computed dynamically per DepthwiseConvUniversalApp's
        # ctor) is the single source of truth for where stage 1's output
        # really lands.
        self._stage1_app = DepthwiseConvUniversalApp(
            inst_path=STAGE1_ASM_PATH,
            input_path=self.input_path,
            kernel=kernel,
            kernel_path=self.kernel_path,
            output_path=None,
            height=ROWS,
            width=COLS,
            channels=channels,
        )

        # Place stage 2's output immediately after stage 1's real output
        # region ends, row-aligned (same convention as both stride2 siblings).
        stage1_output_elements = IN_ROW_GROUPS * channels * CHUNK_ELEMENTS
        self.output_base_addr = (
            self._stage1_app.output_base_addr + stage1_output_elements * 4
        )
        self.output_base_row = self.output_base_addr // ROW_BYTES

    def run(self, *, max_cycles: int = 2_000_000, **kwargs) -> tuple["IpuState", int]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="depthwise_stride2_16_"))
        stage1_bin_path = tmp_dir / "stage1.bin"
        assemble_to_bin_file(STAGE1_ASM_PATH.read_text(), str(stage1_bin_path))

        stage1_app = self._stage1_app
        stage1_app.inst_path = stage1_bin_path
        kwargs.setdefault("state", stage1_app.make_state())
        state, cycles1 = stage1_app.run(max_cycles=max_cycles, **kwargs)

        stage2_bin_path = tmp_dir / "stage2.bin"
        assemble_to_bin_file(STAGE2_ASM_PATH.read_text(), str(stage2_bin_path))

        def stage2_setup(s: "IpuState") -> None:
            # Stage 1 left program_counter at its halted sentinel; reset it
            # so run_until_complete's `while not state.is_halted` loop
            # actually executes stage 2's freshly loaded program instead of
            # exiting immediately with 0 cycles.
            s.program_counter = 0
            # Stage 1 partitioned CR15 for its own mask-shift scheme (cols
            # lanes/group). Stage 2 uses no masks and needs the FULL 128-lane
            # MULT_RES/ACC.STRIDE/ACTIVATE.QUANTIZE view -- reset to no
            # partitioning, all 128 lanes active.
            s.set_cr_dstructure(valid_elements=128, partition=Partition.P0)
            s.regfile.set_cr(3, stage1_app.output_base_row)
            s.regfile.set_cr(4, self.channels)
            s.regfile.set_cr(5, self.output_base_row)
            s.regfile.set_cr(6, self.num_channel_pairs)

        state2, cycles2 = run_test(
            inst_path=stage2_bin_path,
            setup=stage2_setup,
            teardown=lambda s: None,
            max_cycles=max_cycles,
            state=state,
        )

        if self.output_path is not None:
            total_elements = self.num_channel_pairs * CHUNK_ELEMENTS
            raw = state2.xmem.read_address(
                self.output_base_row * ROW_BYTES, total_elements * 4
            )
            # raw is packed [channel_pair, 2, OUT_ROWS, OUT_COLS] -- the two
            # channels in a pair occupy r_acc[0:64] / r_acc[64:128]
            # respectively, and within each channel's 64 elements the layout
            # is OUT_ROWS*OUT_COLS row-major (ACC.STRIDE's row/col decimation
            # preserves row-major order within each 32-element call, and the
            # two calls per channel are row-group 0 then row-group 1).
            # output_path holds the TRUE [channels, OUT_ROWS, OUT_COLS]
            # tensor (see class docstring's file-layout contract).
            arr = np.frombuffer(raw, dtype=np.float32).reshape(
                self.num_channel_pairs, 2, OUT_ROWS, OUT_COLS,
            )
            out = np.empty((self.channels, OUT_ROWS, OUT_COLS), dtype=np.float32)
            out[0::2] = arr[:, 0]
            out[1::2] = arr[:, 1]
            Path(self.output_path).write_bytes(out.tobytes())

        return state2, cycles1 + cycles2


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
    if q.stride != 2:
        return no(f"handles only stride=2; got {q.stride}")
    if not q.is_depthwise:
        return no(f"handles only depthwise (groups==in_channels); got groups={q.groups}, in_channels={q.in_channels}")
    if q.out_channels != q.in_channels:
        return no(f"depthwise requires out_channels==in_channels; got {q.out_channels} vs {q.in_channels}")
    if q.apply_relu or q.has_bias:
        return no("bias/ReLU are not supported by this kernel")
    if (q.height, q.width) != (ROWS, COLS):
        return no(
            f"handles only height=width=16 (the degenerate shape "
            f"depthwise_conv_stride2_narrow can't represent); got "
            f"{q.height}x{q.width}"
        )
    if q.in_channels % 2 != 0:
        return no(f"channels must be even (two channels pack per output chunk); got {q.in_channels}")
    return yes()


def _build(**params):
    q = conv_query(**params)
    return {"channels": q.in_channels}


def _explain(**params):
    return (
        "kernel_size=3, groups=in_channels (depthwise), stride=2, padding=1, "
        "height=width=16: two-stage depthwise stride-2 conv (FP32) for the "
        "degenerate shape whose true output is only half a 128-element "
        "chunk; packs 2 channels per output chunk instead of padding."
    )


def _caveats(**params):
    return (
        "FP32 wide-vector debug mode only (wide_vector_debug=True). This "
        "kernel has no INT8/quantized variant.",
        "Fixed shape only (height=width=16); see depthwise_conv_stride2_narrow "
        "for width in {16,32,64} at other heights.",
        "Not cycle-optimized: unfused two-stage composition (full-res conv "
        "then a separate decimation pass).",
    )


SPEC = KernelSpec(
    name="depthwise_conv_stride2_16",
    op="conv2d",
    variant="depthwise_stride2_16",
    app_class=DepthwiseConvStride2_16App,
    asm="decimate_stage2.asm",
    requires=REQUIRES,
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: conv_query(**params).bundle,
    cost=lambda **params: 0.0,
)
