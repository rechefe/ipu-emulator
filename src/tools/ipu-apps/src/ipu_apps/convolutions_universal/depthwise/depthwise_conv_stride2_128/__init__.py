"""Depthwise 3x3 stride-2 convolution, cols=128 (no packing): 128xNxC -> 64x(N/2)xC.

FP32 wide-vector mode. Two-stage design (built on TOP of the already-verified
depthwise_conv_universal rather than a fresh from-scratch pipeline, per the
project's row-addressing migration playbook -- the base app's fused 9-cyc/ch
pipeline is too tightly timed to safely retrofit ACC.STRIDE decimation into):

  Stage 1: run depthwise_conv_universal's OWN asm, completely UNMODIFIED,
  against the full-resolution input at cols=128. Its harness only supports
  cols in {16,32,64} because its mask-shift scheme needs a CR15 partition
  that divides 128 lanes into >1 packed-row group; at cols=128 there is
  exactly ONE packed row per chunk (no packing at all), which is precisely
  what ``Partition.P0`` (no partitioning) represents -- so this stage is a
  thin subclass that only relaxes the cols=128 special case, not a copy of
  the app's logic. It produces a full-width (no stride) depthwise conv
  result, chunk-interleaved, at its own OUTPUT_BASE_ROW.

  Stage 2 (``decimate_stage2.asm``, new, but deliberately simple: no taps,
  no masks, no r_cyclic, just XMEM loads + ACC.STRIDE + a store): reads
  ROW PAIRS (2j, 2j+1) from stage 1's output and column-decimates each
  128->64 via ACC.STRIDE (which reads MULT_RES, not R_ACC, hence the
  identity-MULT passthrough), packing the pair into one 128-element output
  chunk. Reading only rows 2j/2j+1 and skipping 2j+2.. per iteration IS the
  vertical stride-2 decimation -- stage 1 already computed every row, so
  skipping the odd ones is free (no separate vertical-stride logic needed).

Not cycle-optimized: this is deliberately the lower-risk, unfused
composition. A single-pass fused version would need substantially more
design/debugging investment for a modest cycle-count win.

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_128 import (
        DepthwiseConvStride2_128App,
    )

    app = DepthwiseConvStride2_128App(
        input_path="input.bin",     # raw [channels, rows, 128] float32
        kernel=weights,              # np.ndarray [channels, 3, 3] float32
        output_path="output.bin",
        rows=128, channels=8,
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


class _Stage1FullWidthApp(DepthwiseConvUniversalApp):
    """depthwise_conv_universal with the cols=128 (no-packing) case allowed.

    At cols=128 there's exactly one packed spatial row per 128-element chunk --
    the ``Partition.P0`` (no partitioning) case the base app's mask-shift
    scheme doesn't enumerate because it was written for cols in {16,32,64}
    (multi-row-per-chunk packing). This subclass overrides ONLY the two
    validation/partition-selection points; the base app's asm and all its
    addressing/pipelining logic are untouched.
    """

    def __init__(self, **kwargs) -> None:
        width = kwargs.pop("width", 128)
        if width != 128:
            raise ValueError(f"_Stage1FullWidthApp is for width=128 only, got {width}")
        # The base class picks cols via next_valid_cols(width); width=128
        # already lands on cols=128, so no bypass trick is needed here (the
        # narrow-mode {16,32,64} restriction the old INT8 version worked
        # around no longer applies -- next_valid_cols supports 128 directly).
        super().__init__(width=128, **kwargs)

    def setup(self, state: "IpuState") -> None:
        super().setup(state)
        # Override CR15 dstructure: 128 lanes, ONE group (no packing). The
        # base class's cols_to_partition already maps 128 -> Partition.P0,
        # so this override is now redundant but kept explicit for clarity.
        state.set_cr_dstructure(valid_elements=128, partition=Partition.P0)


class DepthwiseConvStride2_128App(IpuApp):
    """Depthwise 3x3 stride-2 conv, cols=128, FP32 (two-stage, see module doc).

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor -- see
    ``pointwise_conv_unified``'s class docstring for the exact file-layout
    contract this mirrors.

    Args:
        input_path:   Path to input image binary, raw ``[channels, rows,
                      128]`` float32.
        kernel:       Numpy weights of shape ``[channels, 3, 3]`` float32.
        kernel_path:  Alternative: path to a raw ``[channels, 9]`` contiguous
                      float32 file.
        output_path:  Optional path to write output (raw ``[channels,
                      rows/2, 64]`` float32).
        rows:         Spatial height (must be even, out_rows = rows/2 must
                      also be even to pair 2 output rows per chunk).
        channels:     Number of channels (>= 1).
    """

    SELF_ASSEMBLES = True

    def __init__(
        self,
        *,
        rows: int,
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

        if rows < 4 or rows % 2 != 0:
            raise ValueError(f"rows must be even and >= 4, got {rows}")
        out_rows = rows // 2
        if out_rows % 2 != 0:
            raise ValueError(
                f"rows/2 (out_rows={out_rows}) must be even to pair 2 output "
                f"rows per chunk; got rows={rows}"
            )
        if channels < 1:
            raise ValueError(f"channels ({channels}) must be >= 1")

        self.rows = rows
        self.channels = channels
        self.out_rows = out_rows
        self.num_row_pairs = out_rows // 2

        # Build stage 1 now (paths are fixed up with real files in run(), but
        # __init__ needs no real file to exist -- IpuApp.__init__ only stores
        # Path objects). This is the SAME instance run() actually executes,
        # so its region layout (computed dynamically per DepthwiseConvUniversalApp's
        # ctor) is the single source of truth for where stage 1's output
        # really lands.
        self._stage1_app = _Stage1FullWidthApp(
            inst_path=STAGE1_ASM_PATH,
            input_path=self.input_path,
            kernel=kernel,
            kernel_path=self.kernel_path,
            output_path=None,
            height=rows,
            width=128,
            channels=channels,
        )

        # Place stage 2's output immediately after stage 1's real output
        # region ends, row-aligned, so it can never overlap stage 1's
        # still-unread tail chunks regardless of rows*channels.
        stage1_output_elements = rows * channels * CHUNK_ELEMENTS
        self.output_base_addr = (
            self._stage1_app.output_base_addr + stage1_output_elements * 4
        )
        self.output_base_row = self.output_base_addr // ROW_BYTES

    def run(self, *, max_cycles: int = 2_000_000, **kwargs) -> tuple["IpuState", int]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="depthwise_stride2_128_"))
        stage1_bin_path = tmp_dir / "stage1.bin"
        assemble_to_bin_file(STAGE1_ASM_PATH.read_text(), str(stage1_bin_path))

        stage1_app = self._stage1_app
        stage1_app.inst_path = stage1_bin_path
        stage1_app.input_path = self.input_path
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
            s.regfile.set_cr(3, stage1_app.output_base_row)
            s.regfile.set_cr(4, self.channels)
            s.regfile.set_cr(5, self.output_base_row)
            s.regfile.set_cr(6, self.num_row_pairs)

        state2, cycles2 = run_test(
            inst_path=stage2_bin_path,
            setup=stage2_setup,
            teardown=lambda s: None,
            max_cycles=max_cycles,
            state=state,
        )

        if self.output_path is not None:
            total_outputs = self.num_row_pairs * self.channels
            total_elements = total_outputs * CHUNK_ELEMENTS
            raw = state2.xmem.read_address(
                self.output_base_row * ROW_BYTES, total_elements * 4
            )
            # raw is packed [row_pair, channel, 2, 64] -- output_path holds
            # the TRUE [channels, out_rows, 64] tensor (see class docstring's
            # file-layout contract).
            arr = np.frombuffer(raw, dtype=np.float32).reshape(
                self.num_row_pairs, self.channels, 2, 64,
            )
            out = np.empty((self.channels, self.out_rows, 64), dtype=np.float32)
            for rp in range(self.num_row_pairs):
                out[:, 2 * rp, :] = arr[rp, :, 0, :]
                out[:, 2 * rp + 1, :] = arr[rp, :, 1, :]
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
    if q.width != 128:
        return no(f"handles only width=128 (see depthwise_conv_stride2_narrow for cols in 16/32/64); got {q.width}")
    if q.height < 4 or q.height % 4 != 0:
        return no(f"height must be a multiple of 4 (>= 4) so out_rows/2 pairs evenly; got {q.height}")
    return yes()


def _build(**params):
    q = conv_query(**params)
    return {"rows": q.height, "channels": q.in_channels}


def _explain(**params):
    return (
        "kernel_size=3, groups=in_channels (depthwise), stride=2, padding=1, "
        "width=128: two-stage depthwise stride-2 conv (FP32), full-res conv "
        "then column+row decimation."
    )


def _caveats(**params):
    return (
        "FP32 wide-vector debug mode only (wide_vector_debug=True). This "
        "kernel has no INT8/quantized variant.",
        "Not cycle-optimized: unfused two-stage composition (full-res conv "
        "then a separate decimation pass).",
    )


SPEC = KernelSpec(
    name="depthwise_conv_stride2_128",
    op="conv2d",
    variant="depthwise_stride2_128",
    app_class=DepthwiseConvStride2_128App,
    asm="depthwise_conv_universal/depthwise_conv_universal.asm",
    requires=REQUIRES,
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: conv_query(**params).bundle,
    cost=lambda **params: 0.0,
)
