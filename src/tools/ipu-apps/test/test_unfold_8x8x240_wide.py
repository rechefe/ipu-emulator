"""Wide-vector FP32 end-to-end test for unfold_8x8x240.

Runs the REAL kernel binary against a numpy reference in wide-vector debug mode.
No checked-in golden: FP32 inputs are generated here and the expected stride-2
decimation is computed directly.

Unfold is pure data movement (the multiply is by 1.0), so the reference is an
indexing expression, not arithmetic -- any mismatch is a real layout bug.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold_8x8x240 import (
    Unfold8x8x240App, H, W, C, N_STREAMS, N_OUT, N_TOK, LANES,
    DST_BASE, OUTPUT_ROW_BYTES,
)
from ipu_apps.unfold_8x8x240.gen_debug_data import pack_stripe

_INST_BIN = Path(os.environ["UNFOLD_8X8X240_INST_BIN"])


def _run(tmp_path: Path):
    """Run the kernel on seeded input; return (x, cropped_output, state)."""
    rng = np.random.RandomState(0x058)
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)

    input_path = tmp_path / "input_fp32.bin"
    input_path.write_bytes(pack_stripe(x).tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = Unfold8x8x240App(
        inst_path=_INST_BIN,
        input_path=input_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    return x, raw, output_path


def test_unfold_8x8x240_wide_fp32(tmp_path: Path) -> None:
    """Each stream must be the exact stride-2 phase of the input."""
    x, raw, _ = _run(tmp_path)

    assert raw.size == N_STREAMS * N_OUT * N_TOK, (
        f"output has {raw.size} floats, expected {N_STREAMS * N_OUT * N_TOK}"
    )
    got = raw.reshape(N_STREAMS, N_OUT, N_TOK)

    # The four streams are a stride-2 space-to-depth decimation, NOT four
    # contiguous quadrants: stream s takes every other row and column at phase
    # (s // 2, s % 2), i.e. the standard stride-2 convolution decomposition.
    for s in range(N_STREAMS):
        r_ph, c_ph = s // 2, s % 2
        expected = x[:, r_ph::2, c_ph::2].reshape(C, N_TOK)
        np.testing.assert_allclose(
            got[s], expected,
            rtol=1e-4, atol=1e-3,
            err_msg=f"unfold stream {s} (phase {r_ph},{c_ph}) mismatch",
        )


def test_output_shape_and_stale_lanes(tmp_path: Path) -> None:
    """Pin the row contract: one channel per row, 16 valid tokens, stale tail.

    A stream contributes only N_TOK=16 of a row's 128 lanes, and rows are never
    shared between channels. This test reads the UNCROPPED rows the kernel
    actually stored and asserts both halves of that contract: the valid prefix
    matches the reference, and the stale tail is modelled explicitly rather than
    assumed. If a future change ever packed multiple channels into one row, the
    stride assertion here fails.
    """
    x, raw, output_path = _run(tmp_path)

    # teardown() writes the raw uncropped rows alongside the cropped output.
    rows_path = output_path.with_suffix(".rows.bin")
    rows = np.frombuffer(rows_path.read_bytes(), dtype=np.float32)
    assert rows.size == N_STREAMS * N_OUT * LANES, (
        f"raw rows have {rows.size} floats, expected {N_STREAMS * N_OUT * LANES}"
    )
    rows = rows.reshape(N_STREAMS, N_OUT, LANES)

    # One channel per row: each channel occupies a WHOLE 512-byte row.
    assert OUTPUT_ROW_BYTES == LANES * 4
    assert N_TOK * 4 < OUTPUT_ROW_BYTES, "a row must hold one channel, not several"

    for s in range(N_STREAMS):
        r_ph, c_ph = s // 2, s % 2
        expected = x[:, r_ph::2, c_ph::2].reshape(C, N_TOK)

        # Valid prefix: lanes 0..15.
        np.testing.assert_allclose(
            rows[s, :, :N_TOK], expected,
            rtol=1e-4, atol=1e-3,
            err_msg=f"stream {s} valid prefix mismatch",
        )

        # Stale tail: lanes 16..127 are NOT part of the contract's payload.
        # Model them explicitly. ACC.STRIDE writes 32 elements into slot 0, so
        # lanes 16..31 receive the decimation of the input's zero padding
        # (lanes 64..127 of the source row, which the harness zeroes) and are
        # therefore exactly 0.0. Lanes 32..127 of r_acc are never written by
        # this kernel and stay 0.0 from reset.
        stale = rows[s, :, N_TOK:]
        np.testing.assert_array_equal(
            stale, np.zeros_like(stale),
            err_msg=(
                f"stream {s}: stale lanes are not the modelled zeros -- the "
                "input padding or r_acc slot usage changed"
            ),
        )
