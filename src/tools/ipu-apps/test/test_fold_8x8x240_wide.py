"""Wide-vector FP32 end-to-end test for fold_8x8x240.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and the expected
spatial reconstruction is computed directly.

Fold is pure data movement (the multiply is by 1.0), so the reference is an
indexing expression, not arithmetic -- any mismatch is a real layout bug.

The strongest test is the round trip: build a random spatial tensor, run it
through the REAL unfold_8x8x240 kernel (via its documented
``pack_input_rows`` input-prep step), feed unfold's RAW uncropped row output
into fold, and assert we get the original tensor back bit-for-bit (within
FP32 tolerance).

Unlike L3/L4, fold_8x8x240's OUTPUT is naive row-major [H, W] spatial layout
(lane = row*8 + col), NOT a permutation-matched mirror of unfold's own INPUT
packing (``_ROW_PACK_ORDER``) -- see fold_8x8x240.asm's header for why. The
reference here is therefore the plain, un-permuted spatial tensor.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.fold_8x8x240 import Fold8x8x240App
from ipu_apps.unfold.unfold_8x8x240 import (
    Unfold8x8x240App, H, W, C, N_STREAMS, N_OUT, N_TOK, LANES, pack_input_rows,
)

FOLD_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/unfold/fold_8x8x240/fold_8x8x240.asm"
)
UNFOLD_ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/unfold/unfold_8x8x240/unfold_8x8x240.asm"
)


@pytest.fixture(scope="module")
def fold_inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fold_8x8x240.bin"
        assemble_to_bin_file(FOLD_ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


@pytest.fixture(scope="module")
def unfold_inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "unfold_8x8x240.bin"
        assemble_to_bin_file(UNFOLD_ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def _run_unfold_raw_rows(unfold_inst_file: Path, packed: np.ndarray, tmp_path: Path) -> np.ndarray:
    """Run the real unfold_8x8x240 kernel; return its RAW UNCROPPED row output.

    Fold's natural input is unfold's raw ``.rows.bin`` sibling (every stream
    row is a whole 512-byte XMEM row, only the first N_TOK lanes valid), not
    unfold's cropped ``[N_STREAMS, N_OUT, N_TOK]`` convenience array -- see
    fold_8x8x240.asm's header and gen_debug_data.py.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "unfold_input_fp32.bin"
    input_path.write_bytes(packed.tobytes())
    output_path = tmp_path / "unfold_output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = Unfold8x8x240App(
        inst_path=unfold_inst_file,
        input_path=input_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    rows_path = output_path.with_suffix(".rows.bin")
    raw = np.frombuffer(rows_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_STREAMS * N_OUT * LANES
    return raw.reshape(N_STREAMS * N_OUT, LANES)


def _run_fold(fold_inst_file: Path, streams: np.ndarray, tmp_path: Path, *, app_class=Fold8x8x240App):
    """Run fold_8x8x240 on a ``[N_STREAMS * N_OUT, LANES]`` raw-row stream array.

    Returns the ``[C, LANES]`` output array (one row per channel, naive
    row-major spatial order in the first 64 lanes).
    """
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
    assert raw.size == C * LANES, (
        f"output has {raw.size} floats, expected {C * LANES}"
    )
    return raw.reshape(C, LANES)


def test_fold_8x8x240_round_trip(
    fold_inst_file: Path, unfold_inst_file: Path, tmp_path: Path
) -> None:
    """fold(unfold(pack_input_rows(x))) == x, bit-for-bit (mod FP32 rounding from the x1.0 multiply)."""
    rng = np.random.RandomState(0x058)
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)
    packed = pack_input_rows(x)

    unfold_streams = _run_unfold_raw_rows(unfold_inst_file, packed, tmp_path / "unfold")
    got = _run_fold(fold_inst_file, unfold_streams, tmp_path / "fold")

    got_spatial = got[:, : H * W].reshape(C, H, W)
    np.testing.assert_allclose(
        got_spatial, x,
        rtol=1e-6, atol=1e-6,
        err_msg="fold(unfold(x)) != x -- round trip mismatch",
    )


def test_fold_8x8x240_round_trip_poisoned_r_acc(
    fold_inst_file: Path, unfold_inst_file: Path, tmp_path: Path
) -> None:
    """Same round trip, but r_acc starts full of a non-zero sentinel.

    ACC.RESHAPE (unlike ACC.STRIDE) only updates the R_ACC indexes it is
    told to write and leaves the rest alone. This kernel relies on its 8
    (stream, call) ACC.RESHAPE calls per channel partitioning all 64 valid
    r_acc lanes exactly once with no gaps -- if that claim were wrong, stale
    sentinel bytes would leak into the output's valid [0:64) region here.
    """

    class _PoisonedFold(Fold8x8x240App):
        def setup(self, state):
            super().setup(state)
            state.regfile.set_r_acc_bytes(bytearray(b"\xAA" * 512))

    rng = np.random.RandomState(0x059)
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)
    packed = pack_input_rows(x)

    unfold_streams = _run_unfold_raw_rows(unfold_inst_file, packed, tmp_path / "unfold")
    got = _run_fold(
        fold_inst_file, unfold_streams, tmp_path / "fold", app_class=_PoisonedFold
    )

    got_spatial = got[:, : H * W].reshape(C, H, W)
    np.testing.assert_allclose(
        got_spatial, x,
        rtol=1e-6, atol=1e-6,
        err_msg="fold(unfold(x)) != x with poisoned r_acc -- ACC.RESHAPE partition is incomplete",
    )


def test_fold_8x8x240_direct_reference(fold_inst_file: Path, tmp_path: Path) -> None:
    """Hand-computed reference: build the 4 streams directly (no unfold dependency).

    Stream s takes phase (s // 2, s % 2) of the 2x2 spatial decimation (same
    convention unfold_8x8x240's own test documents): TL/TR/BL/BR name the
    phase within each 2x2 block, not a corner of the image. Only the first
    N_TOK lanes of each stream row are populated; the rest are poisoned with
    NaN to confirm fold never reads them (the CR0/CR1-hardwire class of bug
    that bit fold_16x16x192 would not show up here since fold_8x8x240's
    non-zero destination base already uses cr13, not cr0/cr1 -- see the .asm
    header -- but this test still exercises the full source/dest index-table
    arithmetic directly, independent of unfold's own correctness).
    """
    rng = np.random.RandomState(0x05A)
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)

    streams = np.zeros((N_STREAMS * N_OUT, LANES), dtype=np.float32)
    streams[:, N_TOK:] = np.nan
    for s in range(N_STREAMS):
        r_ph, c_ph = s // 2, s % 2
        decimated = x[:, r_ph::2, c_ph::2].reshape(C, N_TOK)   # [C, 16] row-major (dr*4+dc)
        streams[s * N_OUT : (s + 1) * N_OUT, :N_TOK] = decimated

    got = _run_fold(fold_inst_file, streams, tmp_path)

    got_spatial = got[:, : H * W].reshape(C, H, W)
    np.testing.assert_allclose(
        got_spatial, x,
        rtol=1e-6, atol=1e-6,
        err_msg="fold(hand-built streams) != original spatial tensor",
    )
