"""Tests for the wide (W>=384) standard 3x3 convolution app, stride 1.

Runtime-generates random INT8 weights and inputs in the channel-interleaved
-per-spatial-row layout, runs the emulator, and compares against a numpy
reference computed with ipu_mult/ipu_add for bit-exact parity.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.convolutions_universal.conv.conv_universal_wide384 import (
    ConvUniversalWide384App,
)


def _pack_input_wide384(input_chw: np.ndarray, width: int) -> bytes:
    """Pack [in_ch, rows, width] int8 input into the wide384 channel-per-row layout.

    Layout (per the app's addressing spec): each channel's spatial row spans
    `cpr = width // 128` consecutive 128-byte XMEM rows; rows are
    channel-interleaved:
      row(r, ic, cc) = (r * in_channels + ic) * cpr + cc
    """
    in_ch, rows, w = input_chw.shape
    assert w == width
    cpr = width // 128
    packed = bytearray(rows * in_ch * cpr * 128)
    for r in range(rows):
        for ic in range(in_ch):
            row_base = (r * in_ch + ic) * cpr * 128
            for cc in range(cpr):
                for lane in range(128):
                    c = cc * 128 + lane
                    packed[row_base + cc * 128 + lane] = np.uint8(
                        input_chw[ic, r, c]
                    ).item()
    return bytes(packed)


def reference_conv_wide384(
    weights: np.ndarray,
    input_chw: np.ndarray,
    rows: int,
    width: int,
) -> bytes:
    """IPU-math-accurate reference, zero-padded 3x3 conv, stride 1, INT8.

    Produces bytes in the same channel-interleaved-per-row output layout the
    asm writes: out_row(r, f, cc) = (r * out_ch + f) * cpr + cc.
    """
    out_ch, in_ch, _, _ = weights.shape
    cpr = width // 128
    output = bytearray(rows * out_ch * cpr * 128)
    for r in range(rows):
        for f in range(out_ch):
            row_base = (r * out_ch + f) * cpr * 128
            for c in range(width):
                acc: int = 0
                for ic in range(in_ch):
                    for dr in range(3):
                        for dc in range(3):
                            nr, nc = r + dr - 1, c + dc - 1
                            if 0 <= nr < rows and 0 <= nc < width:
                                a = int(weights[f, ic, dr, dc])
                                b = int(input_chw[ic, nr, nc])
                                prod = ipu_mult(a, b, DType.INT8)
                                acc = ipu_add(acc, prod, DType.INT8)
                clamped = max(-128, min(127, acc))
                cc = c // 128
                lane = c % 128
                output[row_base + cc * 128 + lane] = clamped & 0xFF
    return bytes(output)


class TestConvUniversalWide384:

    @pytest.mark.parametrize(
        "width,rows,in_ch,out_ch",
        [
            (384, 8, 1, 2),
            (384, 16, 2, 2),
            (384, 8, 3, 4),
        ],
    )
    def test_conv(
        self,
        tmp_path: Path,
        width: int,
        rows: int,
        in_ch: int,
        out_ch: int,
    ) -> None:
        rng = np.random.RandomState(42 + in_ch * 7 + out_ch + width)
        weights = rng.randint(-32, 33, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
        input_chw = rng.randint(-32, 33, size=(in_ch, rows, width), dtype=np.int8)

        input_packed = _pack_input_wide384(input_chw, width)
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_packed)

        app = ConvUniversalWide384App(
            input_path=input_file,
            kernel=weights,
            output_path=None,
            width=width,
            rows=rows,
            in_channels=in_ch,
            out_channels=out_ch,
        )

        cpr = width // 128
        max_cyc = 200 * rows * out_ch * cpr * in_ch * 9 + 50_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_bytes = rows * out_ch * cpr * 128
        actual = state.xmem.read_address(app.output_base_addr, total_bytes)
        expected = reference_conv_wide384(weights, input_chw, rows, width)

        assert len(actual) == len(expected)

        mismatches = []
        for i in range(len(expected)):
            a_val = struct.unpack_from("b", actual, i)[0]
            e_val = struct.unpack_from("b", expected, i)[0]
            if a_val != e_val:
                lane = i % 128
                row_and_cc = i // 128
                cc = row_and_cc % cpr
                f_and_r = row_and_cc // cpr
                f = f_and_r % out_ch
                r = f_and_r // out_ch
                c = cc * 128 + lane
                mismatches.append(
                    f"  r={r} f={f} cc={cc} c={c} got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )

    @pytest.mark.parametrize(
        "width,rows,in_ch,out_ch",
        [
            (384, 8, 1, 2),
            (384, 16, 2, 2),
        ],
    )
    def test_same_asm_runs_in_wide_vector_debug_mode(
        self,
        tmp_path: Path,
        width: int,
        rows: int,
        in_ch: int,
        out_ch: int,
    ) -> None:
        """The SAME assembled binary must also be correct in wide-vector debug mode.

        See test_conv_universal.py's twin test for the rationale: XMEM
        operands are row numbers and r_cyclic operands are element indices,
        neither depending on 1 B/element (narrow) vs 4 B/element (wide)
        width. Only a single ACTIVATE.QUANTIZE runs per output element (no
        intermediate reload chain like conv_first_layer), so
        wide_vector_quantize_output=True is safe here.
        """
        rng = np.random.RandomState(42 + in_ch * 7 + out_ch + width)
        weights = rng.randint(-8, 9, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
        input_chw = rng.randint(-8, 9, size=(in_ch, rows, width), dtype=np.int8)

        # setup() widens the on-disk (narrow, 1 B/element) input itself once
        # it knows the active mode -- unlike conv_universal's test fixture,
        # the file here must stay in the plain narrow layout.
        input_file = tmp_path / "input_wide.bin"
        input_file.write_bytes(_pack_input_wide384(input_chw, width))

        app = ConvUniversalWide384App(
            input_path=input_file,
            kernel=weights,
            output_path=None,
            width=width,
            rows=rows,
            in_channels=in_ch,
            out_channels=out_ch,
        )
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.INT32,
            wide_vector_quantize_output=True,
        )
        cpr = width // 128
        max_cyc = 200 * rows * out_ch * cpr * in_ch * 9 + 50_000
        state, cycles = app.run(max_cycles=max_cyc, state=state)
        assert cycles > 0

        expected = reference_conv_wide384(weights, input_chw, rows, width)

        mismatches = []
        for i in range(rows * out_ch * cpr):
            actual = state.xmem.read_address((app.output_base_row + i) * 512, 128)
            want = expected[i * 128:(i + 1) * 128]
            for j in range(128):
                if actual[j] != want[j]:
                    a_val = struct.unpack_from("b", actual, j)[0]
                    e_val = struct.unpack_from("b", want, j)[0]
                    mismatches.append(f"  out_row={i} elem={j} got={a_val} expected={e_val}")
        assert not mismatches, (
            f"{len(mismatches)} wide-mode mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )

    def test_odd_out_channels_rejected(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(bytes(1 * 8 * 384))
        weights = np.zeros((3, 1, 3, 3), dtype=np.int8)
        with pytest.raises(ValueError):
            ConvUniversalWide384App(
                input_path=input_file,
                kernel=weights,
                output_path=None,
                width=384,
                rows=8,
                in_channels=1,
                out_channels=3,
            )

    def test_width_512_bonus(self, tmp_path: Path) -> None:
        width, rows, in_ch, out_ch = 512, 4, 1, 2
        rng = np.random.RandomState(999)
        weights = rng.randint(-32, 33, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
        input_chw = rng.randint(-32, 33, size=(in_ch, rows, width), dtype=np.int8)

        input_packed = _pack_input_wide384(input_chw, width)
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_packed)

        app = ConvUniversalWide384App(
            input_path=input_file,
            kernel=weights,
            output_path=None,
            width=width,
            rows=rows,
            in_channels=in_ch,
            out_channels=out_ch,
        )

        cpr = width // 128
        max_cyc = 200 * rows * out_ch * cpr * in_ch * 9 + 50_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_bytes = rows * out_ch * cpr * 128
        actual = state.xmem.read_address(app.output_base_addr, total_bytes)
        expected = reference_conv_wide384(weights, input_chw, rows, width)

        mismatches = []
        for i in range(len(expected)):
            a_val = struct.unpack_from("b", actual, i)[0]
            e_val = struct.unpack_from("b", expected, i)[0]
            if a_val != e_val:
                mismatches.append(f"  i={i} got={a_val} expected={e_val}")
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )
