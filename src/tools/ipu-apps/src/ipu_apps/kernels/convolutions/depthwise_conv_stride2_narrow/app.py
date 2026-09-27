"""Depthwise 3x3 stride-2 convolution, cols in {16, 32, 64} (packed rows):
rows x cols x C -> (rows/2) x (cols/2) x C. FP32 wide-vector mode.

Two-stage design, like the cols=128 sibling (``depthwise_conv_stride2_128``):
composed from ``depthwise_conv_universal``.

  Stage 1: run ``depthwise_conv_universal``'s OWN asm, completely UNMODIFIED,
  against the full-resolution input, through the base harness itself (its
  mask-shift/partition scheme handles packed multi-row-per-chunk layouts
  at cols in {16,32,64}). It produces a full-width (no stride)
  depthwise conv result, chunk-interleaved, at its own OUTPUT_BASE_ROW.

  Stage 2 (this kernel's own ``depthwise_conv_stride2_narrow.asm`` --
  Jinja-templated on ``elements_in_row``): structurally different from
  stride2_128's stage 2 because of the packed layout. At cols=128 each chunk
  holds exactly ONE spatial row (rows_per_chunk=1). At cols in {16,32,64}, ``rows_per_chunk =
  128/cols`` spatial rows are packed into ONE chunk (local_row = row %
  rows_per_chunk, chunk_row_group = row // rows_per_chunk, offset within
  chunk = local_row * cols) -- this is the exact layout
  ``depthwise_conv_universal`` uses.

  The key structural fact that makes stage 2 simple: running
  ``ACC.STRIDE(elements_in_row=cols, horizontal=on, vertical=on, offset)`` on
  ONE loaded stage-1 chunk decimates BOTH dimensions in a single instruction.
  ACC.STRIDE treats the 128-lane MULT_RES as ``rows_per_chunk`` rows of
  ``cols`` elements -- exactly the packed local-row layout -- so horizontal
  stride keeps every other column within each local row, and vertical stride
  keeps every other LOCAL row. Because ``rows_per_chunk`` is always a power
  of 2 and every chunk starts at local_row=0, local-row parity equals
  absolute-row parity, so "every other local row" IS "every other absolute
  row" for free -- no separate row-selection logic is needed, and (critically)
  stage 2 never needs to reach INTO a stage-1 chunk at a non-zero local-row
  offset: it always loads a WHOLE stage-1 chunk and lets ACC.STRIDE's
  built-in row decimation pick out the even local rows.

  Output size per ACC.STRIDE call is (rows_per_chunk/2) * (cols/2) = 32
  elements, ALWAYS (128/4), independent of cols. Since ACC.STRIDE's `offset`
  writes to R_ACC[(offset%4)*32 : +32], exactly 4 stage-1 chunks (at the SAME
  channel, at consecutive absolute row-groups g, g+1, g+2, g+3) combine via 4
  ACC.STRIDE calls (offsets 0,1,2,3) to fill one output chunk. This holds
  uniformly across cols in {16,32,64} -- one clean design serves all three
  native cols values via the single Jinja parameter ``elements_in_row``.

Under the registry, stage 1 is assembled from the sibling
``depthwise_conv_universal.asm`` and stage 2 is rendered from this kernel's
``.asm`` with ``elements_in_row = cols`` at run time; the bound ``inst_path``
(the ``.asm`` assembled standalone, i.e. with its template default 32) is
not run.

Usage (normally through the registry: ``create_harness(
"depthwise_conv_stride2_narrow", params=..., bindings=...)``)::

    from ipu_apps.kernels.convolutions.depthwise_conv_stride2_narrow.app import (
        DepthwiseConvStride2NarrowApp,
    )

    app = DepthwiseConvStride2NarrowApp(
        inst_path="depthwise_conv_stride2_narrow.bin",  # ignored: stage 2 is rendered per cols
        input_path="input.bin",     # raw [channels, rows, cols] float32
        kernel_path="kernel.bin",   # raw [channels, 1, 3, 3] float32 (or kernel=array)
        output_path="output.bin",
        rows=16, cols=32, channels=8,
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

from ipu_apps.kernel_registry.base import IpuApp
from ipu_apps.kernels.convolutions.depthwise_conv_universal.app import (
    DepthwiseConvUniversalApp,
)
from ipu_apps.kernels.convolutions.app import (
    CHUNK_ELEMENTS,
    kernel_asm,
    render_asm,
    universal_spec,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState


ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode


class DepthwiseConvStride2NarrowApp(IpuApp):
    """Depthwise 3x3 stride-2 conv, cols in {16, 32, 64}, FP32 (two-stage).

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor -- see
    ``pointwise_conv_unified``'s class docstring for the exact file-layout
    contract this mirrors.

    Args:
        input_path:   Path to input image binary, raw ``[channels, rows,
                      cols]`` float32.
        kernel:       Numpy weights of shape ``[channels, 3, 3]`` float32.
        kernel_path:  Alternative: path to a raw ``[channels, 9]`` contiguous
                      float32 file.
        output_path:  Optional path to write output (raw ``[channels,
                      rows/2, cols/2]`` float32).
        rows:         Spatial height (must be even; also must be a multiple
                      of 4*rows_per_chunk so stage 2's 4-chunks-per-output-
                      chunk grouping divides evenly -- see module doc).
        cols:         Spatial width (16, 32, or 64) -- native
                      depthwise_conv_universal range.
        channels:     Number of channels (>= 1).
    """

    SELF_ASSEMBLES = True

    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        channels: int,
        kernel: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        # The bound inst_path is kept but not run: stage 1 is assembled from
        # depthwise_conv_universal's .asm and stage 2 is rendered per cols in run().
        kwargs.setdefault("inst_path", "depthwise_conv_stride2_narrow.asm")
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

        kernel_path = getattr(self, "kernel_path", None)
        if kernel is not None and kernel_path is not None:
            raise ValueError("Provide exactly one of kernel= or kernel_path=")
        if kernel is None and kernel_path is None:
            raise ValueError("Provide one of kernel= or kernel_path=")
        self.kernel_path = Path(kernel_path) if kernel_path is not None else None

        valid_cols = {16, 32, 64}
        if cols not in valid_cols:
            raise ValueError(f"cols must be in {valid_cols}, got {cols}")
        rows_per_chunk = 128 // cols
        if rows < 4 or rows % 2 != 0:
            raise ValueError(f"rows must be even and >= 4, got {rows}")
        # Stage 2 groups 4 consecutive stage-1 row-groups (chunks) per
        # channel into one output chunk (see module doc: each ACC.STRIDE
        # call decimates one stage-1 chunk into 32 of the 128 output
        # elements). in_row_groups = rows/rows_per_chunk must be a multiple
        # of 4 for this grouping to divide evenly.
        in_row_groups = rows // rows_per_chunk
        if rows % rows_per_chunk != 0 or in_row_groups % 4 != 0:
            raise ValueError(
                f"rows ({rows}) must be a multiple of 4*rows_per_chunk "
                f"(rows_per_chunk={rows_per_chunk} at cols={cols}); "
                f"need rows % {4 * rows_per_chunk} == 0"
            )
        if channels < 1:
            raise ValueError(f"channels ({channels}) must be >= 1")

        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.rows_per_chunk = rows_per_chunk
        self.in_row_groups = in_row_groups
        self.out_cols = cols // 2
        self.out_rows = rows // 2
        self.out_rows_per_chunk = 128 // self.out_cols
        self.num_out_groups = self.out_rows // self.out_rows_per_chunk

        # Build stage 1 now (paths are fixed up with real files in run(), but
        # __init__ needs no real file to exist -- IpuApp.__init__ only stores
        # Path objects). This is the SAME instance run() actually executes,
        # so its region layout (computed dynamically per DepthwiseConvUniversalApp's
        # ctor) is the single source of truth for where stage 1's output
        # really lands.
        self._stage1_app = DepthwiseConvUniversalApp(
            inst_path="depthwise_conv_universal.asm",  # replaced in run()
            input_path=self.input_path,
            kernel=kernel,
            kernel_path=self.kernel_path,
            output_path=None,
            height=rows,
            width=cols,
            channels=channels,
        )

        # Place stage 2's output immediately after stage 1's real output
        # region ends, row-aligned (same convention as the cols=128 sibling).
        stage1_output_elements = self.in_row_groups * channels * CHUNK_ELEMENTS
        self.output_base_addr = (
            self._stage1_app.output_base_addr + stage1_output_elements * 4
        )
        self.output_base_row = self.output_base_addr // ROW_BYTES

    def run(self, *, max_cycles: int = 2_000_000, **kwargs) -> tuple["IpuState", int]:
        with tempfile.TemporaryDirectory(prefix="depthwise_stride2_narrow_") as tmp:
            return self._run(Path(tmp), max_cycles=max_cycles, **kwargs)

    def _run(self, tmp_dir: Path, *, max_cycles: int, **kwargs) -> tuple["IpuState", int]:
        stage1_bin_path = tmp_dir / "stage1.bin"
        assemble_to_bin_file(kernel_asm("depthwise_conv_universal"), str(stage1_bin_path))

        stage1_app = self._stage1_app
        stage1_app.inst_path = stage1_bin_path
        kwargs.setdefault("state", self.make_state())
        state, cycles1 = stage1_app.run(max_cycles=max_cycles, **kwargs)

        stage2_src = render_asm(
            kernel_asm("depthwise_conv_stride2_narrow"), elements_in_row=self.cols,
        )
        stage2_bin_path = tmp_dir / "stage2.bin"
        assemble_to_bin_file(stage2_src, str(stage2_bin_path))

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
            s.regfile.set_cr(6, self.num_out_groups)

        state2, cycles2 = run_test(
            inst_path=stage2_bin_path,
            setup=stage2_setup,
            teardown=lambda s: None,
            max_cycles=max_cycles,
            state=state,
        )

        if self.output_path is not None:
            total_outputs = self.num_out_groups * self.channels
            total_elements = total_outputs * CHUNK_ELEMENTS
            raw = state2.xmem.read_address(
                self.output_base_row * ROW_BYTES, total_elements * 4
            )
            # raw is packed [out_group, channel, out_rows_per_chunk, out_cols]
            # -- output_path holds the TRUE [channels, out_rows, out_cols]
            # tensor (see class docstring's file-layout contract).
            arr = np.frombuffer(raw, dtype=np.float32).reshape(
                self.num_out_groups, self.channels, self.out_rows_per_chunk, self.out_cols,
            )
            out = np.empty((self.channels, self.out_rows, self.out_cols), dtype=np.float32)
            for og in range(self.num_out_groups):
                for local_row in range(self.out_rows_per_chunk):
                    orow = og * self.out_rows_per_chunk + local_row
                    out[:, orow, :] = arr[og, :, local_row, :]
            Path(self.output_path).write_bytes(out.tobytes())

        return state2, cycles1 + cycles2


# -- registry declaration ---------------------------------------------------


def _supports(q):
    if q.kernel_size != 3:
        return f"handles only kernel_size=3; got {q.kernel_size}"
    if q.dilation != 1:
        return f"handles only dilation=1; got {q.dilation}"
    if q.padding != 1:
        return f"handles only padding=1; got {q.padding}"
    if q.stride != 2:
        return f"handles only stride=2; got {q.stride}"
    if not q.is_depthwise:
        return (f"handles only depthwise (groups==in_channels); got groups={q.groups}, "
                f"in_channels={q.in_channels}")
    if q.out_channels != q.in_channels:
        return (f"depthwise requires out_channels==in_channels; got {q.out_channels} "
                f"vs {q.in_channels}")
    if q.apply_relu or q.has_bias:
        return "bias/ReLU are not supported by this kernel"
    if q.width not in (16, 32, 64):
        return (f"handles only width in (16, 32, 64) (see depthwise_conv_stride2_128 for "
                f"width=128); got {q.width}")
    rows_per_chunk = 128 // q.width
    if q.height < 4 or q.height % 4 != 0 or (q.height // rows_per_chunk) % 4 != 0:
        return (
            f"height ({q.height}) must be a multiple of 4*rows_per_chunk "
            f"(rows_per_chunk={rows_per_chunk} at width={q.width})"
        )
    return None


SPEC = universal_spec(
    DepthwiseConvStride2NarrowApp,
    variant="depthwise_stride2_narrow",
    supports=_supports,
    build=lambda q: {"rows": q.height, "cols": q.width, "channels": q.in_channels},
    explain=lambda q: (
        "kernel_size=3, groups=in_channels (depthwise), stride=2, padding=1, "
        "width in {16,32,64}: two-stage depthwise stride-2 conv (FP32), "
        "full-res conv then joint row+col decimation."
    ),
    caveats=lambda q: (
        "Not cycle-optimized: unfused two-stage composition (full-res conv "
        "then a separate decimation pass).",
    ),
)
