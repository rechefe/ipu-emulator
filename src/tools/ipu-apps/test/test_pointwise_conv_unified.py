"""Pytest wrapper for the pointwise_conv_unified correctness suite.

Reuses the standalone runner's reference + config list (which live next to the
app, in ``pointwise_conv_unified/test_unified.py``) and parametrizes them so the
configs run as individual pytest cases. The asm is assembled once per session.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified import (
    PointwiseConvUnifiedApp,
)
from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified.test_unified import (
    ASM_PATH,
    TEST_CONFIGS,
    pack_input,
    reference_pointwise,
    run_one,
)


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("pw_unified")
    inst_file = tmp / "pointwise_conv_unified.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
    return inst_file


@pytest.mark.parametrize("rows,cols,in_ch,out_ch", TEST_CONFIGS)
def test_pointwise_conv_unified(
    inst_file: Path, rows: int, cols: int, in_ch: int, out_ch: int
) -> None:
    cycles, mismatches, actual, expected = run_one(
        inst_file, rows, cols, in_ch, out_ch
    )
    assert cycles > 0
    assert mismatches == 0, (
        f"{mismatches} mismatches for {rows}x{cols} ic={in_ch} oc={out_ch}\n"
        f"  first OC actual: {actual[0, 0, :8]}\n"
        f"  first OC expect: {expected[0, 0, :8]}"
    )


class TestPointwiseConvUnifiedWideVectorDebug:
    """The SAME assembled binary must also be correct in wide-vector debug mode.

    See test_conv_universal.py's twin test for the rationale: XMEM operands
    are row numbers and r_cyclic operands are element indices, neither
    depending on 1 B/element (narrow) vs 4 B/element (wide) width.
    """

    @pytest.mark.parametrize(
        "rows,cols,in_ch,out_ch",
        [
            (16, 16, 8, 8),      # single pass, single row-group
            (16, 16, 144, 8),    # multi-pass (1 full + tail)
            (32, 32, 128, 16),   # multiple row-groups
        ],
    )
    def test_same_asm_runs_in_wide_vector_debug_mode(
        self,
        inst_file: Path,
        rows: int,
        cols: int,
        in_ch: int,
        out_ch: int,
    ) -> None:
        rng = np.random.RandomState(42 + in_ch * 7 + out_ch + rows + cols)
        weights = rng.randint(-3, 4, size=(out_ch, in_ch), dtype=np.int8)
        input_chw = rng.randint(-3, 4, size=(in_ch, rows, cols), dtype=np.int8)

        narrow_input = pack_input(input_chw, rows, cols)
        packed = bytearray(len(narrow_input) * 4)
        for i, byte in enumerate(narrow_input):
            struct.pack_into(
                "<i", packed, i * 4, struct.unpack_from("b", narrow_input, i)[0]
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_file = tmp / "input_wide.bin"
            kernel_file = tmp / "kernel.bin"
            input_file.write_bytes(bytes(packed))
            kernel_file.write_bytes(
                weights.reshape(out_ch * in_ch).view(np.uint8).tobytes()
            )

            app = PointwiseConvUnifiedApp(
                inst_path=inst_file,
                input_path=input_file,
                kernel_path=kernel_file,
                output_path=None,
                dtype="INT8",
                rows=rows, cols=cols, in_channels=in_ch, out_channels=out_ch,
            )
            state = IpuState(
                wide_vector_debug=True,
                wide_vector_arithmetic=WideVectorArithmetic.INT32,
                wide_vector_quantize_output=True,
            )
            max_cyc = 50 * in_ch * out_ch * (rows * cols // 128) + 100_000
            state, cycles = app.run(max_cycles=max_cyc, state=state)
            assert cycles > 0

            expected = reference_pointwise(weights, input_chw)

            row_groups = (rows * cols) // 128
            rows_per_chunk = 128 // cols
            mismatches = []
            for rg in range(row_groups):
                for oc in range(out_ch):
                    i = rg * out_ch + oc
                    actual = state.xmem.read_address((app.output_base_row + i) * 512, 128)
                    block = np.frombuffer(actual, dtype=np.uint8)[
                        :rows_per_chunk * cols
                    ].view(np.int8).reshape(rows_per_chunk, cols)
                    want = expected[oc, rg * rows_per_chunk:(rg + 1) * rows_per_chunk, :]
                    if not np.array_equal(block, want):
                        mismatches.append(
                            f"  rg={rg} oc={oc} got={block.flatten()[:8]} "
                            f"expected={want.flatten()[:8]}"
                        )
            assert not mismatches, (
                f"{len(mismatches)} wide-mode mismatches (first 20):\n"
                + "\n".join(mismatches[:20])
            )
