"""Self-contained tests for the universal pointwise (1x1) convolution.

Tests multiple spatial/channel configurations including the grouped path
(when in_channels does not divide 128) and the fast path (when it does).
Compares emulator output against ipu_math-accurate reference.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.pointwise.pointwise_conv_universal import (
    PointwiseConvUniversalApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_ROW_BYTES,
)


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "pointwise" / "pointwise_conv_universal"
    / "pointwise_conv_universal.asm"
)


def reference_pointwise_conv_universal(
    input_bytes: bytes,
    kernel_bytes: bytes,
    rows: int,
    cols: int,
    in_channels: int,
    out_channels: int,
) -> bytes:
    """IPU-math-accurate reference for universal pointwise (1x1) convolution.

    Input layout (interleaved by row-group and channel):
      IC ic, row-group rg: offset = rg * in_channels * 128 + ic * 128
      Each 128-byte chunk holds one row-group's worth of elements for one IC.

    Kernel layout: kernel[oc * in_channels + ic]

    Output layout (INT8, interleaved by row-group and OC):
      OC oc, row-group rg: offset = (rg * out_channels + oc) * 128
    """
    dtype = DType.INT8
    row_groups = (rows * cols) // 128
    output = bytearray(row_groups * out_channels * 128)

    for rg in range(row_groups):
        for oc in range(out_channels):
            for elem in range(128):
                acc: int = 0
                for ic in range(in_channels):
                    ki = oc * in_channels + ic
                    bi = rg * in_channels * 128 + ic * 128 + elem
                    a = kernel_bytes[ki]
                    b = input_bytes[bi]
                    prod = ipu_mult(a, b, dtype)
                    acc = ipu_add(acc, prod, dtype)
                clamped = max(-128, min(127, acc))
                out_idx = (rg * out_channels + oc) * 128 + elem
                output[out_idx] = clamped & 0xFF

    return bytes(output)


def _gen_test_data(
    rows: int, cols: int, in_channels: int, out_channels: int, seed: int = 42,
) -> tuple[bytes, bytes]:
    """Generate random INT8 input and kernel in the correct memory layout."""
    rng = np.random.RandomState(seed)
    row_groups = (rows * cols) // 128

    input_raw = rng.randint(-4, 5, size=row_groups * in_channels * 128, dtype=np.int8)
    kernel_raw = rng.randint(-4, 5, size=out_channels * in_channels, dtype=np.int8)

    return input_raw.view(np.uint8).tobytes(), kernel_raw.view(np.uint8).tobytes()


class TestPointwiseConvUniversal:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        tmp = tmp_path_factory.mktemp("pw_univ")
        inst_file = tmp / "pointwise_conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "rows,cols,in_ch,out_ch",
        [
            (16, 16, 16, 16),   # fast path: in_ch divides 128
            (16, 16, 32, 16),   # fast path: in_ch divides 128
            (32, 32, 16, 32),   # fast path, larger spatial
            (16, 16, 96, 32),   # grouped path: G=32, num_groups=3
            (32, 32, 160, 32),  # grouped path: G=32, num_groups=5
        ],
    )
    def test_pointwise(
        self,
        inst_file: Path,
        tmp_path: Path,
        rows: int,
        cols: int,
        in_ch: int,
        out_ch: int,
    ) -> None:
        input_bytes, kernel_bytes = _gen_test_data(rows, cols, in_ch, out_ch)

        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_bytes)
        kernel_file.write_bytes(kernel_bytes)

        app = PointwiseConvUniversalApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            dtype="INT8",
            rows=rows,
            cols=cols,
            in_channels=in_ch,
            out_channels=out_ch,
        )

        row_groups = (rows * cols) // 128
        max_cyc = 5_000 * row_groups * out_ch * in_ch + 100_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_bytes = row_groups * out_ch * OUTPUT_ROW_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_pointwise_conv_universal(
            input_bytes, kernel_bytes, rows, cols, in_ch, out_ch,
        )

        assert len(actual) == len(expected), (
            f"Output size mismatch: {len(actual)} vs {len(expected)}"
        )

        mismatches = []
        for i in range(len(expected)):
            a_val = struct.unpack_from("b", actual, i)[0]
            e_val = struct.unpack_from("b", expected, i)[0]
            if a_val != e_val:
                rg = i // (out_ch * 128)
                rem = i % (out_ch * 128)
                oc = rem // 128
                elem = rem % 128
                mismatches.append(
                    f"  rg={rg} oc={oc} elem={elem} "
                    f"got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )
