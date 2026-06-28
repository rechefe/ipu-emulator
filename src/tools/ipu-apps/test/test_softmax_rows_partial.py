"""Correctness tests for packed partial-width row-softmax.

Covers the working regimes (P in {1,2,4} for all chunk counts; P=8 up to 2
chunks) against a numpy softmax reference, plus the guard that rejects the
known-broken P=8 >=3-chunk case (see STATUS.md).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_rows_partial import (
    SoftmaxRowsPartialApp,
    OUTPUT_BASE_ADDR,
    CHUNK_BYTES,
    partition_size,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/softmax/softmax_rows_partial/softmax_rows_partial.asm"
)


def _reference(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def _run(inst_file: Path, x: np.ndarray, n: int) -> np.ndarray:
    rows = x.shape[0]
    ps = partition_size(n)
    P = 128 // ps
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "input.bin"
        inp.write_bytes(x.astype(np.float32).tobytes())
        app = SoftmaxRowsPartialApp(
            inst_path=inst_file, input_path=inp, output_path=None, n=n, rows=rows
        )
        state, _ = app.run(max_cycles=2_000_000)
        out = np.zeros((rows, n), np.float32)
        for r in range(rows):
            base = OUTPUT_BASE_ADDR + (r // P) * CHUNK_BYTES + (r % P) * ps * 4
            out[r] = np.frombuffer(state.xmem.read_address(base, n * 4), dtype=np.float32)
    return out


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "softmax_rows_partial.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


@pytest.mark.parametrize(
    "n,rows,seed",
    [
        # P=1 (N 65..128)
        (128, 4, 0), (100, 6, 1), (65, 3, 2),
        # P=2 (N 33..64)
        (64, 4, 3), (50, 10, 4), (33, 8, 5),
        # P=4 (N 17..32), incl. multi-chunk
        (32, 8, 6), (32, 12, 7), (20, 12, 8), (17, 16, 9),
        # P=8 (N 1..16), up to 2 chunks (the working range)
        (16, 8, 10), (16, 16, 11), (8, 8, 12), (8, 16, 13), (1, 8, 14),
    ],
)
def test_partial_softmax_matches_numpy(inst_file, n, rows, seed):
    rng = np.random.RandomState(seed)
    x = (rng.randn(rows, n) * 3.0).astype(np.float32)
    out = _run(inst_file, x, n)
    ref = _reference(x)
    assert np.abs(out - ref).max() < 1e-4
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_padding_rows_ignored(inst_file):
    """Rows not a multiple of P are zero-padded internally; real rows correct."""
    n, rows = 32, 6  # P=4, padded to 8
    rng = np.random.RandomState(20)
    x = (rng.randn(rows, n) * 3.0).astype(np.float32)
    out = _run(inst_file, x, n)
    ref = _reference(x)
    assert np.abs(out - ref).max() < 1e-4


def test_p8_three_chunks_guarded():
    """The known-broken P=8 >=3-chunk regime must raise, not return garbage."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "i.bin"
        inp.write_bytes(np.zeros((24, 8), np.float32).tobytes())
        with pytest.raises(NotImplementedError, match="P=8"):
            SoftmaxRowsPartialApp(
                inst_path=tmp + "/x", input_path=inp, output_path=None, n=8, rows=24
            )
