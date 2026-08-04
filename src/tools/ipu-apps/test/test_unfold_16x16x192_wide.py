"""Wide-vector FP32 end-to-end test for unfold_16x16x192.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and the expected
sub-grid rearrangement is computed directly.

Unfold is pure data movement (the multiply is by 1.0), so the reference is an
indexing expression, not arithmetic -- any mismatch is a real layout bug.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold_16x16x192 import (
    Unfold16x16x192App, H, W, C, N_STRIPES, N_STREAMS, N_OUT, N_TOK, LANES,
)

_INST_BIN = Path(os.environ["UNFOLD_16X16X192_INST_BIN"])

_STRIPE_H = H // N_STRIPES      # 8 spatial rows per stripe


def test_unfold_16x16x192_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x016)

    # Spatial tensor [C, H, W] in FP32.
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)

    # NHCW-striped input: row (stripe, ch) holds that channel's 8 spatial rows
    # x 16 cols, flattened row-major, one XMEM row each.
    src = np.zeros((N_STRIPES * C, LANES), dtype=np.float32)
    for stripe in range(N_STRIPES):
        r0 = stripe * _STRIPE_H
        for ch in range(C):
            block = x[ch, r0 : r0 + _STRIPE_H, :]        # [8, 16]
            src[stripe * C + ch, : _STRIPE_H * W] = block.reshape(-1)

    input_path = tmp_path / "input_fp32.bin"
    input_path.write_bytes(src.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = Unfold16x16x192App(
        inst_path=_INST_BIN,
        input_path=input_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_STREAMS * N_OUT * LANES, (
        f"output has {raw.size} floats, expected {N_STREAMS * N_OUT * LANES}"
    )
    got = raw.reshape(N_STREAMS, N_OUT, LANES)

    # The four streams are a stride-2 space-to-depth decimation, NOT four
    # contiguous quadrants: stream s takes every other row and column at phase
    # (s // 2, s % 2), i.e. the standard stride-2 convolution decomposition.
    # TL/TR/BL/BR name the phase within each 2x2 block, not a corner of the image.
    for s in range(N_STREAMS):
        r_ph, c_ph = s // 2, s % 2
        expected = x[:, r_ph::2, c_ph::2].reshape(C, N_TOK)
        # Only the first N_TOK lanes are valid; the rest are stale r_acc lanes
        # that STR_ACC_REG always writes (documented in the module docstring).
        np.testing.assert_allclose(
            got[s, :, :N_TOK], expected,
            rtol=1e-6, atol=1e-6,
            err_msg=f"unfold stream {s} (phase {r_ph},{c_ph}) mismatch",
        )
