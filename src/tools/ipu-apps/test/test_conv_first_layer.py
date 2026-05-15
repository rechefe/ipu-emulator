"""Self-contained test for first-layer conv: 256x256x3 -> 128x128x16, stride 2.

Generates random input/kernel, assembles, runs, and compares against
a Python reference that matches the IPU's arithmetic and boundary behavior.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.conv.conv_first_layer import (
    ConvFirstLayerApp,
    OUTPUT_BASE_ADDR,
    IN_ROWS,
    IN_COLS,
    IN_CHANNELS,
    OUT_ROWS,
    OUT_COLS,
    OUT_CHANNELS,
    CHANNEL_STRIDE,
)


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "conv" / "conv_first_layer"
    / "conv_first_layer.asm"
)


def reference_conv_first_layer(
    input_bytes: bytes,
    kernel_bytes: bytes,
) -> bytes:
    """Compute reference stride-2 3x3 conv matching the assembly's behavior.

    Input layout: CHW (channel c, row r, col c_pos at c*65536 + r*256 + c_pos).
    Kernel layout: filter f, channel c, tap t at f*27 + c*9 + t.
    Tap order per channel: (kr=-1,kc=-1), ..., (kr=+1,kc=+1).

    Output layout: interleaved 128-byte chunks.
      Row r, filter f: byte offset = (r * 16 + f) * 128 + output_col.

    The assembly processes left half (cols 0..127) and right half (cols 128..255)
    independently. At the boundary, kc=-1 at right-half position 0 (col 128)
    is masked to zero. This means output col 64 is missing the kc=-1
    contribution from col 127. The reference matches this behavior.
    """
    dtype = DType.INT8
    output = bytearray(OUT_ROWS * OUT_CHANNELS * OUT_COLS)

    for f in range(OUT_CHANNELS):
        for out_r in range(OUT_ROWS):
            center_r = out_r * 2

            for out_c in range(OUT_COLS):
                center_c = out_c * 2

                # Determine which half this output col belongs to
                # Left half processes cols 0..127, stride-2 even -> output cols 0..63
                # Right half processes cols 128..255, stride-2 even -> output cols 64..127
                if out_c < 64:
                    half_start = 0
                else:
                    half_start = 128

                acc: int = 0
                for c in range(IN_CHANNELS):
                    for kr in range(-1, 2):
                        for kc in range(-1, 2):
                            ir = center_r + kr
                            ic = center_c + kc

                            # Top border: row -1 is zero-padded
                            if ir < 0 or ir >= IN_ROWS:
                                continue

                            # Left border: kc=-1 at position 0 of each half
                            if ic < half_start:
                                continue

                            # Right border: kc=+1 at position 127 of each half
                            if ic >= half_start + 128:
                                continue

                            # Read input byte (CHW layout)
                            in_idx = c * CHANNEL_STRIDE + ir * IN_COLS + ic
                            b = input_bytes[in_idx]

                            # Read kernel byte
                            tap = (kr + 1) * 3 + (kc + 1)
                            ki = f * 27 + c * 9 + tap
                            a = kernel_bytes[ki]

                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)

                # First quantization (aaq after conv)
                val_q = max(-128, min(127, acc))

                # The stride step does identity multiply + acc.stride + aaq.
                # Since val_q is already in INT8 range, the second aaq is a no-op.
                # Output byte position
                out_byte = (out_r * OUT_CHANNELS + f) * OUT_COLS + out_c
                output[out_byte] = val_q & 0xFF

    return bytes(output)


def _gen_test_data(seed: int = 42) -> tuple[bytes, bytes]:
    """Generate random INT8 input and kernel data with small values."""
    rng = np.random.RandomState(seed)
    input_size = IN_CHANNELS * IN_ROWS * IN_COLS  # 3 * 256 * 256
    kernel_size = OUT_CHANNELS * IN_CHANNELS * 9  # 16 * 3 * 9
    input_data = rng.randint(-3, 4, size=input_size, dtype=np.int8)
    kernel_data = rng.randint(-3, 4, size=kernel_size, dtype=np.int8)
    return input_data.view(np.uint8).tobytes(), kernel_data.view(np.uint8).tobytes()


class TestConvFirstLayer:

    def test_256x256x3_to_128x128x16(self, tmp_path: Path) -> None:
        """Full first-layer conv: 256x256x3 -> 128x128x16, stride 2, INT8."""
        # Generate data
        input_bytes, kernel_bytes = _gen_test_data()
        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_bytes)
        kernel_file.write_bytes(kernel_bytes)

        # Assemble
        inst_file = tmp_path / "prog.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        # Run
        app = ConvFirstLayerApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
        )
        state, cycles = app.run(max_cycles=200_000_000)
        assert cycles > 0

        # Read output from xmem
        total_out_bytes = OUT_ROWS * OUT_CHANNELS * OUT_COLS
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_out_bytes)

        # Compute reference
        expected = reference_conv_first_layer(input_bytes, kernel_bytes)

        assert len(actual) == len(expected), (
            f"Output size mismatch: {len(actual)} vs {len(expected)}"
        )

        # Compare byte-by-byte with diagnostics
        mismatches = []
        for i in range(len(expected)):
            if actual[i] != expected[i]:
                out_row = i // (OUT_CHANNELS * OUT_COLS)
                rem = i % (OUT_CHANNELS * OUT_COLS)
                filt = rem // OUT_COLS
                col = rem % OUT_COLS
                a_val = struct.unpack("b", bytes([actual[i]]))[0]
                e_val = struct.unpack("b", bytes([expected[i]]))[0]
                mismatches.append(
                    f"  byte {i}: row={out_row} filter={filt} col={col} "
                    f"got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )
