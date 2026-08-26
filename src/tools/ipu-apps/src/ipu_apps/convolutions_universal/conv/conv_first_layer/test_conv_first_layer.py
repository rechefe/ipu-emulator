"""Self-contained test for first-layer conv: 256x256x3 -> 128x128x16, stride 2.

Generates random INT8 input/kernel/bias, assembles, runs, and compares against a
Python reference that matches the IPU datapath exactly:

  * per 128-lane slot: r_acc = bias + full 3x3x3 conv for that slot's positions,
    then ACTIVATE.QUANTIZE identity -> INT8 (intermediate clamp);
  * decimate the two slots' even positions (stride 2) into the 128 outputs;
  * ACTIVATE.QUANTIZE relu -> final INT8.

Border: true pad=1.  Output row 0 skips kr=-1 (row -1 out of bounds); the bottom
row (255) is in bounds so the last output row is a normal middle row.  Left col 0
(kc=-1) and right col 255 (kc=+1) are mask-zeroed.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.convolutions_universal.conv.conv_first_layer import (
    ConvFirstLayerApp,
    CHUNK_BYTES,
    OUTPUT_BASE_ADDR,
    OUTPUT_BASE_ROW,
    IN_ROWS,
    IN_COLS,
    IN_CHANNELS,
    OUT_ROWS,
    OUT_COLS,
    OUT_CHANNELS,
    IN_ROW_GROUP,
    OUT_ROW_GROUP,
)

ASM_PATH = Path(__file__).resolve().parent / "conv_first_layer.asm"


def _clamp_int8(v: int) -> int:
    return int(max(-128, min(127, int(v))))


def reference(input_bytes: bytes, kernel: np.ndarray, bias: np.ndarray) -> bytes:
    """Reference matching the asm datapath (intermediate clamp + decimate + relu).

    input_bytes: row-interleaved (row r, ch) at (r*3+ch)*256, int8.
    kernel: [16, 3, 3, 3] int8 (out, in, kh, kw), taps kr=-1..+1, kc=-1..+1.
    bias:   [16] int8.
    Output: row-interleaved (row r, filter f) at (r*16+f)*128, int8.
    """
    x = np.frombuffer(input_bytes, dtype=np.int8).astype(np.int32)
    x = x.reshape(IN_ROWS, IN_CHANNELS, IN_COLS)  # x[r, ch, col]
    k = kernel.astype(np.int32)
    b = bias.astype(np.int32)

    out = bytearray(OUT_ROWS * OUT_ROW_GROUP)
    for f in range(OUT_CHANNELS):
        for orow in range(OUT_ROWS):
            cr = orow * 2
            # full 256-wide accumulation for this (orow, f): position p = bias + conv
            acc256 = np.zeros(IN_COLS, np.int32) + b[f]
            for c in range(IN_CHANNELS):
                for kr in (-1, 0, 1):
                    ir = cr + kr
                    if not (0 <= ir < IN_ROWS):
                        continue
                    strip = x[ir, c]  # 256
                    for kc in (-1, 0, 1):
                        tap = k[f, c, kr + 1, kc + 1]
                        sh = np.roll(strip, -kc)
                        if kc == -1:
                            sh[0] = 0        # col 0 has no left neighbour
                        if kc == +1:
                            sh[IN_COLS - 1] = 0  # col 255 has no right neighbour
                        acc256 += tap * sh
            # intermediate INT8 clamp (identity) applied to BOTH 128-lane slots
            half = np.array([_clamp_int8(v) for v in acc256], np.int32)
            dec = half[0:IN_COLS:2]                 # 128 stride-2 outputs
            final = np.array([_clamp_int8(max(0, v)) for v in dec], np.int32)  # relu + requant
            base = (orow * OUT_CHANNELS + f) * OUT_COLS
            out[base:base + OUT_COLS] = bytes((int(v) & 0xFF) for v in final)
    return bytes(out)


def _gen(seed: int = 42):
    rng = np.random.RandomState(seed)
    x = rng.randint(-3, 4, size=IN_ROWS * IN_CHANNELS * IN_COLS, dtype=np.int8)
    k = rng.randint(-3, 4, size=(OUT_CHANNELS, IN_CHANNELS, 3, 3)).astype(np.int8)
    b = rng.randint(-5, 6, size=OUT_CHANNELS).astype(np.int8)
    return x.view(np.uint8).tobytes(), k, b


def test_256x256x3_to_128x128x16(tmp_path: Path) -> None:
    input_bytes, kernel, bias = _gen()
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(input_bytes)

    inst_file = tmp_path / "prog.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

    app = ConvFirstLayerApp(
        inst_path=inst_file,
        input_path=input_file,
        kernel=kernel,
        bias=bias,
        output_path=None,
    )
    state, cycles = app.run(max_cycles=500_000_000)
    assert cycles > 0

    total = OUT_ROWS * OUT_ROW_GROUP
    actual = bytes(state.xmem.read_address(OUTPUT_BASE_ADDR, total))
    expected = reference(input_bytes, kernel, bias)
    assert len(actual) == len(expected)

    mism = []
    for i in range(len(expected)):
        if actual[i] != expected[i]:
            orow = i // OUT_ROW_GROUP
            rem = i % OUT_ROW_GROUP
            filt = rem // OUT_COLS
            col = rem % OUT_COLS
            a = struct.unpack("b", bytes([actual[i]]))[0]
            e = struct.unpack("b", bytes([expected[i]]))[0]
            mism.append(f"  row={orow} filter={filt} col={col} got={a} exp={e}")
    assert not mism, f"{len(mism)} mismatches (first 20):\n" + "\n".join(mism[:20])


def test_256x256x3_to_128x128x16_wide_vector_debug_mode(tmp_path: Path) -> None:
    """The SAME assembled binary must also be correct in wide-vector debug mode.

    See test_conv_universal.py's twin test for the rationale: XMEM operands
    are row numbers and r_cyclic operands are element indices, neither
    depending on 1 B/element (narrow) vs 4 B/element (wide) width. Here that
    means ``ConvFirstLayerApp.setup()``/``teardown()`` must scale their raw
    ``xmem.write_address``/``read_address`` calls by the active mode's row
    size themselves (unlike CR values, which the emulator already scales).
    """
    input_bytes, kernel, bias = _gen(seed=99)
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(input_bytes)

    inst_file = tmp_path / "prog.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

    app = ConvFirstLayerApp(
        inst_path=inst_file,
        input_path=input_file,
        kernel=kernel,
        bias=bias,
        output_path=None,
    )
    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.INT32,
        wide_vector_quantize_output=False,
    )
    state, cycles = app.run(max_cycles=500_000_000, state=state)
    assert cycles > 0

    total = OUT_ROWS * OUT_ROW_GROUP
    row_bytes = CHUNK_BYTES * 4
    raw = state.xmem.read_address(OUTPUT_BASE_ROW * row_bytes, total * 4)
    actual = bytes(
        (max(-128, min(127, struct.unpack_from("<i", raw, i * 4)[0])) & 0xFF)
        for i in range(total)
    )
    expected = reference(input_bytes, kernel, bias)
    assert len(actual) == len(expected)

    mism = []
    for i in range(len(expected)):
        if actual[i] != expected[i]:
            orow = i // OUT_ROW_GROUP
            rem = i % OUT_ROW_GROUP
            filt = rem // OUT_COLS
            col = rem % OUT_COLS
            a = struct.unpack("b", bytes([actual[i]]))[0]
            e = struct.unpack("b", bytes([expected[i]]))[0]
            mism.append(f"  row={orow} filter={filt} col={col} got={a} exp={e}")
    assert not mism, (
        f"{len(mism)} wide-mode mismatches (first 20):\n" + "\n".join(mism[:20])
    )
