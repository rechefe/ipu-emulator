"""Self-contained tests for the universal standard 3x3 convolution (FPB=14).

Runtime-generates random INT8 weights and inputs, runs the emulator, and
compares against a numpy reference computed with ipu_mult/ipu_add for bit-
exact parity. Exercises full blocks, partial last blocks, cross-chunk
spatial sizes, and a density sanity check.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_universal import (
    ConvUniversalApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
)
from ipu_apps.convolutions_universal.weights import pack_conv_weights_dense


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "conv" / "conv_universal"
    / "conv_universal.asm"
)


def _pack_input_chunked(
    input_chw: np.ndarray,
    rows: int,
    cols: int,
) -> bytes:
    """Pack [in_ch, rows, cols] int8 input into the chunk-interleaved layout.

    Offset formula (mirrors the asm / gen_test_data):
      rows_per_chunk = 128 // cols
      chunk = r // rows_per_chunk
      local_row = r % rows_per_chunk
      offset = (chunk * in_ch + ch) * 128 + local_row * cols + c
    """
    in_ch, ir, ic = input_chw.shape
    assert (ir, ic) == (rows, cols)
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128
    packed = bytearray(num_chunks * in_ch * 128)
    for ch in range(in_ch):
        for r in range(rows):
            for c in range(cols):
                chunk = r // rows_per_chunk
                local_row = r % rows_per_chunk
                offset = (chunk * in_ch + ch) * 128 + local_row * cols + c
                packed[offset] = np.uint8(input_chw[ch, r, c]).item()
    return bytes(packed)


def reference_conv_universal(
    weights: np.ndarray,
    input_chw: np.ndarray,
    rows: int,
    cols: int,
) -> bytes:
    """IPU-math-accurate reference.

    Produces bytes in the same chunk-interleaved output layout the asm writes:
      chunk = r // rows_per_chunk
      local_row = r % rows_per_chunk
      elem = local_row * cols + c
      byte_offset = (chunk * out_ch + f) * 128 + elem

    aaq quantization: clamp int32 accumulator to [-128, 127], store as int8.
    """
    out_ch, in_ch, _, _ = weights.shape
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128
    output = bytearray(num_chunks * out_ch * 128)
    for f in range(out_ch):
        for r in range(rows):
            for c in range(cols):
                acc: int = 0
                for ic in range(in_ch):
                    for dr in range(3):
                        for dc in range(3):
                            nr, nc = r + dr - 1, c + dc - 1
                            if 0 <= nr < rows and 0 <= nc < cols:
                                a = int(weights[f, ic, dr, dc])
                                b = int(input_chw[ic, nr, nc])
                                prod = ipu_mult(a, b, DType.INT8)
                                acc = ipu_add(acc, prod, DType.INT8)
                clamped = max(-128, min(127, acc))
                chunk = r // rows_per_chunk
                local_row = r % rows_per_chunk
                elem = local_row * cols + c
                out_idx = (chunk * out_ch + f) * 128 + elem
                output[out_idx] = clamped & 0xFF
    return bytes(output)


class TestConvUniversal:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        tmp = tmp_path_factory.mktemp("conv_universal")
        inst_file = tmp / "conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "in_ch,out_ch,rows,cols",
        [
            (16, 4, 16, 16),   # partial last block (16 % 14 = 2)
            (14, 4, 16, 16),   # exactly one full block
            (28, 4, 16, 16),   # exactly two full blocks
            (10, 4, 16, 16),   # single partial block (10 < 14)
            (16, 8, 32, 32),   # cross-chunk, multiple filters
        ],
    )
    def test_conv(
        self,
        inst_file: Path,
        tmp_path: Path,
        in_ch: int,
        out_ch: int,
        rows: int,
        cols: int,
    ) -> None:
        rng = np.random.RandomState(42 + in_ch * 7 + out_ch)
        weights = rng.randint(-32, 33, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
        input_chw = rng.randint(-32, 33, size=(in_ch, rows, cols), dtype=np.int8)

        input_packed = _pack_input_chunked(input_chw, rows, cols)
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_packed)

        app = ConvUniversalApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=weights,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=cols,
            in_channels=in_ch, out_channels=out_ch,
        )

        # Rough cycle budget: each output pixel costs ~9 taps * in_ch.
        num_chunks = (rows * cols) // 128
        max_cyc = 2_000 * num_chunks * out_ch * math.ceil(in_ch / 14) + 50_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_bytes = num_chunks * out_ch * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_conv_universal(weights, input_chw, rows, cols)

        assert len(actual) == len(expected)

        mismatches = []
        rows_per_chunk = 128 // cols
        for i in range(len(expected)):
            a_val = struct.unpack_from("b", actual, i)[0]
            e_val = struct.unpack_from("b", expected, i)[0]
            if a_val != e_val:
                elem = i % 128
                f_and_chunk = i // 128
                f = f_and_chunk % out_ch
                chunk = f_and_chunk // out_ch
                local_row = elem // cols
                col = elem % cols
                row = chunk * rows_per_chunk + local_row
                mismatches.append(
                    f"  chunk={chunk} f={f} r={row} c={col} "
                    f"got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )


def test_density_sanity_in_ch_512() -> None:
    """At in_ch=512, dense FPB=14 layout packs to 4736 bytes/filter vs 8192 old."""
    weights = np.zeros((1, 512, 3, 3), dtype=np.int8)
    packed = pack_conv_weights_dense(weights, DType.INT8, kernel_size=3)
    assert len(packed) == math.ceil(512 / 14) * 128 == 4736, (
        f"expected 4736 bytes, got {len(packed)}"
    )
