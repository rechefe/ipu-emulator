"""Wide-vector FP32 end-to-end test for unfold_32x32x144.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and the expected
stride-2 rearrangement is computed directly.

Unfold is pure data movement (the multiply is by 1.0), so the reference is an
indexing expression, not arithmetic -- any mismatch is a real layout bug.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.unfold_32x32x144 import (
    Unfold32x32x144App, H, W, C, N_STRIPES, N_STREAMS, N_OUT, N_TG, LANES,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/unfold/unfold_32x32x144/unfold_32x32x144.asm"
)

_STRIPE_H = H // N_STRIPES      # 4 spatial rows per stripe


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "unfold_32x32x144.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def test_unfold_32x32x144_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
    rng = np.random.RandomState(0x032)

    # Spatial tensor [C, H, W] in FP32.
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)

    # NHCW-striped input: row (stripe, ch) holds that channel's 4 spatial rows
    # x 32 cols, flattened row-major, one XMEM row each.
    src = np.zeros((N_STRIPES * C, LANES), dtype=np.float32)
    for stripe in range(N_STRIPES):
        r0 = stripe * _STRIPE_H
        for ch in range(C):
            block = x[ch, r0 : r0 + _STRIPE_H, :]        # [4, 32]
            src[stripe * C + ch, : _STRIPE_H * W] = block.reshape(-1)

    input_path = tmp_path / "input_fp32.bin"
    input_path.write_bytes(src.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = Unfold32x32x144App(
        inst_path=inst_file,
        input_path=input_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_STREAMS * N_OUT * N_TG * LANES, (
        f"output has {raw.size} floats, "
        f"expected {N_STREAMS * N_OUT * N_TG * LANES}"
    )
    # Rows are (stream, channel, token group); a 16x16 sub-grid is 256 tokens,
    # so each channel's stream spans N_TG groups of LANES lanes.
    got = raw.reshape(N_STREAMS, N_OUT, N_TG * LANES)

    # The four streams are a stride-2 space-to-depth decimation, NOT four
    # contiguous quadrants: stream s takes every other row and column at phase
    # (s // 2, s % 2), i.e. the standard stride-2 convolution decomposition.
    for s in range(N_STREAMS):
        r_ph, c_ph = s // 2, s % 2
        expected = x[:, r_ph::2, c_ph::2].reshape(C, -1)
        np.testing.assert_allclose(
            got[s, :, : expected.shape[1]], expected,
            rtol=1e-6, atol=1e-6,
            err_msg=f"unfold stream {s} (phase {r_ph},{c_ph}) mismatch",
        )
