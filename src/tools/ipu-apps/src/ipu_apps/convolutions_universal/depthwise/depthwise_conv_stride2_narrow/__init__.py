"""Depthwise 3x3 stride-2 convolution, cols in {16, 32, 64} (packed rows):
rows x cols x C -> (rows/2) x (cols/2) x C.

Two-stage design, same philosophy as the cols=128 sibling
(``depthwise_conv_stride2_128``): build on TOP of the already-verified
``depthwise_conv_universal`` rather than a fresh from-scratch pipeline.

  Stage 1: run ``depthwise_conv_universal``'s OWN asm, completely UNMODIFIED,
  against the full-resolution input. Unlike the cols=128 sibling, NO subclass
  is needed here -- cols in {16,32,64} is exactly the base app's native,
  fully-supported range (its mask-shift/partition scheme was built for
  packed multi-row-per-chunk layouts). It produces a full-width (no stride)
  depthwise conv result, chunk-interleaved, at its own OUTPUT_BASE_ROW.

  Stage 2 (``decimate_stage2.asm``, new, Jinja-templated on
  ``elements_in_row``): structurally different from stride2_128's stage 2
  because of the packed layout. At cols=128 each 128-byte XMEM chunk held
  exactly ONE spatial row (rows_per_chunk=1). At cols in {16,32,64},
  ``rows_per_chunk = 128/cols`` spatial rows are packed into ONE chunk
  (local_row = row % rows_per_chunk, chunk_row_group = row // rows_per_chunk,
  byte offset within chunk = local_row * cols) -- this is the exact layout
  ``depthwise_conv_universal`` (and its test's ``reference_depthwise``) uses.

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
  writes to r_acc[(offset%4)*32 : +32], exactly 4 stage-1 chunks (at the SAME
  channel, at consecutive absolute row-groups g, g+1, g+2, g+3) combine via 4
  ACC.STRIDE calls (offsets 0,1,2,3) to fill one 128-byte OUTPUT chunk. This
  holds uniformly across cols in {16,32,64} (see the .asm's docstring for the
  worked arithmetic per cols value) -- one clean design serves all three
  native cols values via the single Jinja parameter ``elements_in_row``.

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_narrow import (
        DepthwiseConvStride2NarrowApp,
    )

    app = DepthwiseConvStride2NarrowApp(
        input_path="input.bin",     # rows * channels * 128 bytes, row-group-interleaved chunks
        kernel_path="kernel.bin",   # channels * 9 bytes, INT8 raw
        output_path="output.bin",
        rows=16, cols=32, channels=8,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.emulator import run_test
from ipu_emu.ipu_config import Partition

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
    CHUNK_BYTES,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState


STAGE1_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "depthwise_conv_universal" / "depthwise_conv_universal.asm"
)
STAGE2_ASM_TEMPLATE_PATH = Path(__file__).resolve().parent / "decimate_stage2.asm"

# Stage 1's own regions run 0x000000..~0x130000+; stay well clear (same
# guard-band convention as the cols=128 sibling).
OUTPUT_BASE_ADDR = 0x180000
OUTPUT_BASE_ROW = OUTPUT_BASE_ADDR // CHUNK_BYTES


def _as_signed_byte(value: int) -> int:
    """Reinterpret a wire byte as the signed INT8 it encodes."""
    v = value & 0xFF
    return v - 256 if v > 127 else v


class DepthwiseConvStride2NarrowApp(IpuApp):
    """Depthwise 3x3 stride-2 conv, cols in {16, 32, 64}, INT8 only (two-stage).

    Args:
        input_path:   Path to input image binary, chunk-interleaved by
                      channel per ``depthwise_conv_universal``'s packed
                      layout: row-group ``r // rows_per_chunk``, channel
                      ``ch`` at chunk index (row_group*channels+ch)*128,
                      local offset (r % rows_per_chunk)*cols + c.
        kernel_path:  Path to kernel binary (channels * 9 bytes, INT8 raw).
        output_path:  Optional path to write output.
        rows:         Spatial height (must be even; also must be a multiple
                      of 4*rows_per_chunk so stage 2's 4-chunks-per-output-
                      chunk grouping divides evenly -- see module doc).
        cols:         Spatial width (16, 32, or 64) -- native
                      depthwise_conv_universal range.
        channels:     Number of channels (>= 1).
    """

    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        channels: int,
        **kwargs,
    ) -> None:
        kwargs.pop("inst_path", None)  # unused; stages assemble their own
        super().__init__(inst_path=STAGE1_ASM_PATH, **kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)

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
        # ctor -- see that class for why the fixed *_BASE_ADDR gaps were
        # replaced) is the single source of truth for where stage 1's output
        # really lands, instead of assuming the stale module-level
        # STAGE1_OUTPUT_BASE_ROW constant (see depthwise_conv_stride2_128's
        # identical fix for the bug this pattern caught there).
        self._stage1_app = DepthwiseConvUniversalApp(
            inst_path=STAGE1_ASM_PATH,
            input_path=self.input_path,
            kernel_path=self.kernel_path,
            output_path=None,
            dtype="INT8",
            rows=rows,
            cols=cols,
            channels=channels,
        )

        # Place stage 2's output immediately after stage 1's real output
        # region ends, row-aligned (same convention as the cols=128 sibling).
        stage1_output_bytes = self.in_row_groups * channels * CHUNK_BYTES
        self.output_base_addr = (
            self._stage1_app.output_base_addr + stage1_output_bytes
        )
        self.output_base_row = self.output_base_addr // CHUNK_BYTES

    def run(self, *, max_cycles: int = 2_000_000, **kwargs) -> tuple["IpuState", int]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="depthwise_stride2_narrow_"))
        stage1_bin_path = tmp_dir / "stage1.bin"
        assemble_to_bin_file(STAGE1_ASM_PATH.read_text(), str(stage1_bin_path))

        stage1_app = self._stage1_app
        stage1_app.inst_path = stage1_bin_path
        state, cycles1 = stage1_app.run(max_cycles=max_cycles, **kwargs)

        stage2_src = jinja2.Template(STAGE2_ASM_TEMPLATE_PATH.read_text()).render(
            elements_in_row=self.cols,
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
            # str_post_aaq_reg always writes a fixed 512-byte POST_AAQ_REG
            # payload per output row (see ipu.py::execute_str_post_aaq_reg),
            # which spans 4 rows in narrow mode (128 B each, INT8-packed) but
            # only 1 row in wide-vector debug mode (512 B, 4 B/element). Read
            # back at the mode's actual row size, then narrow to 1 B/element
            # so ``output_path`` always holds the same canonical INT8 layout.
            total_outputs = self.num_out_groups * self.channels
            element_width = 4 if state2.wide_vector_debug else 1
            row_bytes = CHUNK_BYTES * element_width
            total_elements = total_outputs * CHUNK_BYTES
            raw = state2.xmem.read_address(
                self.output_base_row * row_bytes, total_elements * element_width
            )
            if element_width == 1:
                data = bytes(raw)
            else:
                data = bytes(
                    _as_signed_byte(struct.unpack_from("<i", raw, i * 4)[0]) & 0xFF
                    for i in range(total_elements)
                )
            Path(self.output_path).write_bytes(data)

        return state2, cycles1 + cycles2
