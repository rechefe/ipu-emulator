"""Depthwise 3x3 stride-2 convolution, cols=128 (no packing): 128xNxC -> 64x(N/2)xC.

Two-stage design (built on TOP of the already-verified depthwise_conv_universal
rather than a fresh from-scratch pipeline, per the project's row-addressing
migration playbook -- the base app's fused 9-cyc/ch pipeline is too tightly
timed to safely retrofit ACC.STRIDE decimation into):

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
  identity-MULT passthrough), packing the pair into one 128-byte output
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
        inst_path=None,  # unused; stage binaries are assembled internally
        input_path="input.bin",     # rows * channels * 128 bytes, row-interleaved chunks
        kernel_path="kernel.bin",   # channels * 9 bytes, INT8 raw
        output_path="output.bin",
        rows=128, channels=8,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.emulator import run_test
from ipu_emu.ipu_config import Partition

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import dump_outputs
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
    OUTPUT_BASE_ROW as STAGE1_OUTPUT_BASE_ROW,
    OUTPUT_BASE_ADDR as STAGE1_OUTPUT_BASE_ADDR,
    CHUNK_BYTES,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState


STAGE1_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "depthwise_conv_universal" / "depthwise_conv_universal.asm"
)
STAGE2_ASM_PATH = Path(__file__).resolve().parent / "decimate_stage2.asm"

# Stage 1's own output region starts at STAGE1_OUTPUT_BASE_ADDR and grows by
# rows*channels*128 bytes (chunk-interleaved, one chunk per channel per row
# -- num_chunks == rows at cols=128). A FIXED stage-2 output address (the
# original 0x180000) only gave 0x50000 bytes of headroom (2560 chunks worth)
# before stage 1's own output spilled past it and stage 2's early writes
# clobbered still-unread stage-1 tail chunks. Fixed: compute the stage-2
# base per-instance, in DepthwiseConvStride2_128App.__init__, always placed
# right after stage 1's output region ends (see self.output_base_addr).
_DEFAULT_OUTPUT_BASE_ADDR = 0x180000
OUTPUT_BASE_ADDR = _DEFAULT_OUTPUT_BASE_ADDR  # kept for backwards-compat imports
OUTPUT_BASE_ROW = OUTPUT_BASE_ADDR // CHUNK_BYTES


class _Stage1FullWidthApp(DepthwiseConvUniversalApp):
    """depthwise_conv_universal with the cols=128 (no-packing) case allowed.

    At cols=128 there's exactly one packed spatial row per 128-byte chunk --
    the ``Partition.P0`` (no partitioning) case the base app's mask-shift
    scheme doesn't enumerate because it was written for cols in {16,32,64}
    (multi-row-per-chunk packing). This subclass overrides ONLY the two
    validation/partition-selection points; the base app's asm and all its
    addressing/pipelining logic are untouched.
    """

    def __init__(self, **kwargs) -> None:
        cols = kwargs.pop("cols", 128)
        if cols != 128:
            raise ValueError(f"_Stage1FullWidthApp is for cols=128 only, got {cols}")
        # Bypass the base __init__'s `cols not in {16,32,64}` check by
        # temporarily satisfying it with a placeholder, then fixing self.cols.
        super().__init__(cols=64, **{k: v for k, v in kwargs.items() if k != "rows"},
                          rows=kwargs["rows"])
        self.cols = 128
        # __init__ computed self.num_chunks = rows*cols//128 using the
        # temporary cols=64 view -- HALF the true cols=128 value (a "chunk"
        # is 128/cols packed spatial rows; at cols=128 that's 1 row/chunk, at
        # cols=64 it's 2 rows/chunk). This was the actual remaining bug: the
        # chunk-loop limit (cr11 in setup()) derives from num_chunks, so the
        # loop ran for half as many chunks as it should, silently truncating
        # (and misinterpreting the stride of) the row loop. Recompute here so
        # setup()'s cr11 override further down uses the correct value.
        rows = kwargs["rows"]
        self.num_chunks = (rows * 128) // 128

    def setup(self, state: "IpuState") -> None:
        # Re-run the base setup logic but with Partition.P0 instead of the
        # base's cols_to_partition lookup (which doesn't have a 128 entry).
        # Simplest correct approach: monkeypatch set_cr_dstructure's default
        # by calling super().setup() is not viable (it would raise on
        # cols=128's absence from cols_to_partition), so setup is
        # reimplemented here by delegating to the base class via a saved
        # cols value trick: temporarily present cols=64 for the parts of
        # setup that don't depend on partitioning, then override CR15.
        original_cols = self.cols
        self.cols = 64  # satisfies base setup's cols_to_partition lookup
        super().setup(state)
        self.cols = original_cols
        # Override CR15 dstructure: 128 lanes, ONE group (no packing).
        state.set_cr_dstructure(valid_elements=128, partition=Partition.P0)
        # Re-write the mask blob for the TRUE cols=128 (base's setup wrote
        # it for its temporary cols=64 view, which is wrong for this app).
        from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
            build_border_mask_blob,
        )
        from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
            MASK_BASE_ROW,
        )
        row_bytes = CHUNK_BYTES * self._element_width
        state.xmem.write_address(
            MASK_BASE_ROW * row_bytes, build_border_mask_blob(128),
        )
        # CR4 = cols, read by the asm's `add lr1 lr0 cr4` (lr1 = cols, used
        # only for the kc=-1/+1 walk step `add lr3 lr3 lr1`) -- must be 128.
        state.regfile.set_cr(4, 128)
        # CR14 = 256 - 2*cols - 2 (r_cyclic ELEMENT end-of-9 walking step from
        # tap 9 to the next channel's tap 1). Base setup computed this off the
        # temporary cols=64 view above (126, WRONG); the true cols=128 value
        # is -2 (as unsigned 32-bit: 0xFFFFFFFE). Missing this override was
        # the actual bug the first time -- it silently produced wrong (not
        # faulting) conv results everywhere, since the walk step only shows
        # up as a subtle per-channel kernel-index drift.
        state.regfile.set_cr(14, (256 - 2 * 128 - 2) & 0xFFFFFFFF)


class DepthwiseConvStride2_128App(IpuApp):
    """Depthwise 3x3 stride-2 conv, cols=128, INT8 only (two-stage, see module doc).

    Args:
        input_path:   Path to input image binary, row-interleaved by channel:
                      (row r, channel ch) at (r*channels + ch)*128 bytes.
        kernel_path:  Path to kernel binary (channels * 9 bytes, INT8 raw).
        output_path:  Optional path to write output.
        rows:         Spatial height (must be even, out_rows = rows/2 must
                      also be even to pair 2 output rows per 128-byte chunk).
        channels:     Number of channels (>= 1).
    """

    def __init__(
        self,
        *,
        rows: int,
        channels: int,
        **kwargs,
    ) -> None:
        kwargs.pop("inst_path", None)  # unused; stages assemble their own
        super().__init__(inst_path=STAGE1_ASM_PATH, **kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)

        if rows < 4 or rows % 2 != 0:
            raise ValueError(f"rows must be even and >= 4, got {rows}")
        out_rows = rows // 2
        if out_rows % 2 != 0:
            raise ValueError(
                f"rows/2 (out_rows={out_rows}) must be even to pair 2 output "
                f"rows per 128-byte chunk; got rows={rows}"
            )
        if channels < 1:
            raise ValueError(f"channels ({channels}) must be >= 1")

        self.rows = rows
        self.channels = channels
        self.out_rows = out_rows
        self.num_row_pairs = out_rows // 2

        # Stage 1's output region is [STAGE1_OUTPUT_BASE_ADDR, +rows*channels*128)
        # (num_chunks == rows at cols=128, one 128-byte chunk per channel per
        # row). Place stage 2's output immediately after it, row-aligned, so
        # it can never overlap stage 1's still-unread tail chunks regardless
        # of rows*channels (see the module-level comment on OUTPUT_BASE_ADDR
        # for the bug this replaces -- a fixed 0x180000 base only had 2560
        # chunks of headroom before stage 1's output spilled into it).
        stage1_output_bytes = rows * channels * CHUNK_BYTES
        self.output_base_addr = STAGE1_OUTPUT_BASE_ADDR + stage1_output_bytes
        self.output_base_row = self.output_base_addr // CHUNK_BYTES

    def run(self, *, max_cycles: int = 2_000_000, **kwargs) -> tuple["IpuState", int]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="depthwise_stride2_128_"))
        stage1_bin_path = tmp_dir / "stage1.bin"
        assemble_to_bin_file(STAGE1_ASM_PATH.read_text(), str(stage1_bin_path))

        stage1_app = _Stage1FullWidthApp(
            inst_path=stage1_bin_path,
            input_path=self.input_path,
            kernel_path=self.kernel_path,
            output_path=None,
            dtype="INT8",
            rows=self.rows,
            cols=128,
            channels=self.channels,
        )
        state, cycles1 = stage1_app.run(max_cycles=max_cycles)

        stage2_bin_path = tmp_dir / "stage2.bin"
        assemble_to_bin_file(STAGE2_ASM_PATH.read_text(), str(stage2_bin_path))

        def stage2_setup(s: "IpuState") -> None:
            # Stage 1 left program_counter at its halted sentinel; reset it
            # so run_until_complete's `while not state.is_halted` loop
            # actually executes stage 2's freshly loaded program instead of
            # exiting immediately with 0 cycles.
            s.program_counter = 0
            s.regfile.set_cr(3, STAGE1_OUTPUT_BASE_ROW)
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
            dump_outputs(
                state2, self.output_path,
                self.output_base_addr, CHUNK_BYTES, total_outputs,
            )

        return state2, cycles1 + cycles2
