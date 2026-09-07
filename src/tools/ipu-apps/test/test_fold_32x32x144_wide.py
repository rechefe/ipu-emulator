"""Wide-vector FP32 end-to-end test for fold_32x32x144.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and the expected
spatial reconstruction is computed directly.

Fold is pure data movement (the multiply is by 1.0), so the reference is an
indexing expression, not arithmetic -- any mismatch is a real layout bug.

The strongest test is the round trip: build a random spatial tensor, run it
through the REAL unfold_32x32x144 kernel, feed unfold's output into fold, and
assert we get the original tensor back bit-for-bit (within FP32 tolerance).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.fold_32x32x144 import Fold32x32x144App
from ipu_apps.unfold.unfold_32x32x144 import (
    Unfold32x32x144App, H, W, C, N_STRIPES, N_STREAMS, N_OUT, N_TG, LANES,
)

FOLD_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/unfold/fold_32x32x144/fold_32x32x144.asm"
)
UNFOLD_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/unfold/unfold_32x32x144/unfold_32x32x144.asm"
)

_STRIPE_H = H // N_STRIPES      # 4 spatial rows per stripe


@pytest.fixture(scope="module")
def fold_inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fold_32x32x144.bin"
        assemble_to_bin_file(FOLD_ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


@pytest.fixture(scope="module")
def unfold_inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "unfold_32x32x144.bin"
        assemble_to_bin_file(UNFOLD_ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def _spatial_to_nhcw_striped(x: np.ndarray) -> np.ndarray:
    """[C, H, W] -> NHCW-striped rows (matches unfold_32x32x144's own input recipe)."""
    src = np.zeros((N_STRIPES * C, LANES), dtype=np.float32)
    for stripe in range(N_STRIPES):
        r0 = stripe * _STRIPE_H
        for ch in range(C):
            block = x[ch, r0 : r0 + _STRIPE_H, :]        # [4, 32]
            src[stripe * C + ch, : _STRIPE_H * W] = block.reshape(-1)
    return src


def _run_unfold(unfold_inst_file: Path, striped: np.ndarray, tmp_path: Path) -> np.ndarray:
    """Run the real unfold_32x32x144 kernel; return its [N_STREAMS, N_OUT, N_TG*LANES] output."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "unfold_input_fp32.bin"
    input_path.write_bytes(striped.tobytes())
    output_path = tmp_path / "unfold_output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = Unfold32x32x144App(
        inst_path=unfold_inst_file,
        input_path=input_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_STREAMS * N_OUT * N_TG * LANES
    return raw.reshape(N_STREAMS, N_OUT * N_TG, LANES)


def _run_fold(fold_inst_file: Path, streams: np.ndarray, tmp_path: Path, *, app_class=Fold32x32x144App):
    """Run fold_32x32x144 on a [N_STREAMS, N_OUT*N_TG, LANES] stream array; return the striped output array."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "fold_input_fp32.bin"
    input_path.write_bytes(np.ascontiguousarray(streams, dtype=np.float32).tobytes())
    output_path = tmp_path / "fold_output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = app_class(
        inst_path=fold_inst_file,
        input_path=input_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_STRIPES * C * LANES, (
        f"output has {raw.size} floats, expected {N_STRIPES * C * LANES}"
    )
    return raw.reshape(N_STRIPES * C, LANES)


def test_fold_32x32x144_round_trip(
    fold_inst_file: Path, unfold_inst_file: Path, tmp_path: Path
) -> None:
    """fold(unfold(x)) == x, bit-for-bit (mod FP32 rounding from the x1.0 multiply)."""
    rng = np.random.RandomState(0x032)
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)
    striped = _spatial_to_nhcw_striped(x)

    unfold_streams = _run_unfold(unfold_inst_file, striped, tmp_path / "unfold")
    got_striped = _run_fold(fold_inst_file, unfold_streams, tmp_path / "fold")

    np.testing.assert_allclose(
        got_striped, striped,
        rtol=1e-6, atol=1e-6,
        err_msg="fold(unfold(x)) != x -- round trip mismatch",
    )


def test_fold_32x32x144_round_trip_poisoned_r_acc(
    fold_inst_file: Path, unfold_inst_file: Path, tmp_path: Path
) -> None:
    """Same round trip, but r_acc starts full of a non-zero sentinel.

    ACC.RESHAPE (unlike ACC.STRIDE) only updates the R_ACC indexes it is
    told to write and leaves the rest alone. This kernel relies on its 16
    (stream, call) ACC.RESHAPE calls per stripe partitioning all 128 r_acc
    lanes exactly once with no gaps -- if that claim were wrong, stale
    sentinel bytes would leak into the output here.
    """

    class _PoisonedFold(Fold32x32x144App):
        def setup(self, state):
            super().setup(state)
            state.regfile.set_r_acc_bytes(bytearray(b"\xAA" * 512))

    rng = np.random.RandomState(0x033)
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)
    striped = _spatial_to_nhcw_striped(x)

    unfold_streams = _run_unfold(unfold_inst_file, striped, tmp_path / "unfold")
    got_striped = _run_fold(
        fold_inst_file, unfold_streams, tmp_path / "fold", app_class=_PoisonedFold
    )

    np.testing.assert_allclose(
        got_striped, striped,
        rtol=1e-6, atol=1e-6,
        err_msg="fold(unfold(x)) != x with poisoned r_acc -- ACC.RESHAPE partition is incomplete",
    )


def test_fold_32x32x144_direct_reference(fold_inst_file: Path, tmp_path: Path) -> None:
    """Hand-computed reference: build the 4 streams directly (no unfold dependency).

    Stream s takes phase (s // 2, s % 2) of the 2x2 spatial decimation (same
    convention unfold_32x32x144's own test documents): TL/TR/BL/BR name the
    phase within each 2x2 block, not a corner of the image. Token group tg
    holds decimated rows [tg*8, tg*8+8) of the 16x16 decimated grid,
    row-major. Unlike fold_16x16x192, every lane of every stream row is
    real data here -- there is no stale-padding tail to poison/verify.
    """
    rng = np.random.RandomState(0x034)
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)
    striped = _spatial_to_nhcw_striped(x)

    streams = np.zeros((N_STREAMS, N_OUT * N_TG, LANES), dtype=np.float32)
    for s in range(N_STREAMS):
        r_ph, c_ph = s // 2, s % 2
        decimated = x[:, r_ph::2, c_ph::2].reshape(C, 16, 16)   # [C, 16 dec rows, 16 dec cols]
        for ch in range(C):
            for tg in range(N_TG):
                block = decimated[ch, tg * 8 : tg * 8 + 8, :]   # [8, 16]
                streams[s, ch * N_TG + tg, :] = block.reshape(-1)

    got_striped = _run_fold(fold_inst_file, streams, tmp_path)

    np.testing.assert_allclose(
        got_striped, striped,
        rtol=1e-6, atol=1e-6,
        err_msg="fold(hand-built streams) != original spatial tensor",
    )
