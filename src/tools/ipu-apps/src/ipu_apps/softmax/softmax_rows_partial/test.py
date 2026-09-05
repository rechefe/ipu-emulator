"""Correctness tests for packed partial-width row-softmax.

Covers P=1/2/4/8 against a numpy softmax reference, including high chunk
counts that used to trigger Pass-4 cross-partition contamination (see
STATUS.md) before the per-partition R_MASK fix, and row counts past 128,
which used to overflow MULT.RC.VE's src operand before the group loop.

Tests read the app's OUTPUT FILE rather than XMEM, so they also pin the
round-trip property: output layout == input layout (row-major rows x N).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.softmax.test_support import random_case, reference, run_array
from ipu_apps.kernel_registry.cases import run_case

from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp

ASM_PATH = Path(__file__).with_name("softmax_rows_partial.asm")

def _reference(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def _run(inst_file: Path, x: np.ndarray, n: int) -> np.ndarray:
    _, out = run_array("softmax_rows_partial", inst_file, x, 1)
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


@pytest.mark.parametrize(
    "n,rows",
    [
        (64, 70),   # P=2, num_chunks=35 -- broke before the per-partition mask fix
        (32, 36),   # P=4, num_chunks=9  -- broke before the fix
        (8, 24),    # P=8, num_chunks=3  -- the originally-documented broken case
        (8, 64),    # P=8, num_chunks=8
        (16, 40),   # P=8 (n=16), num_chunks=5
    ],
)
def test_many_chunks_correct(inst_file, n, rows):
    """High chunk counts that used to trigger Pass-4 cross-partition
    contamination (see STATUS.md): each partition's MULT.RC.VE produces a
    full 128-lane result where only [p*ps, p*ps+N) is that partition's own
    data -- the rest is real (not zero) data from OTHER rows via the rc_idx
    wraparound trick. acc.add/acc.add.first used to write/accumulate all 128
    lanes unmasked, so this leaked data corrupted neighboring partitions'
    slots once chunk count was high enough for the leaked values to exceed
    tolerance. Fixed via a per-partition R_MASK slot (mask_offset=p) that
    restricts each MULT.RC.VE to exactly its own lane range, zeroing
    everywhere else -- see _partition_masks() and the .asm's p4_run_p*
    blocks.
    """
    rng = np.random.RandomState(0)
    x = (rng.randn(rows, n) * 5.0).astype(np.float32)
    out = _run(inst_file, x, n)
    ref = _reference(x)
    assert np.abs(out - ref).max() < 1e-4


@pytest.mark.parametrize(
    "n,rows",
    [
        (128, 129),   # P=1: one row past a full group
        (128, 300),   # P=1: two full groups + a short one
        (100, 257),   # P=1, N<ps
        (64, 130),    # P=2: one row past a full group (128 rows = 64 chunks)
        (64, 1000),   # P=2, many groups
        (32, 300),    # P=4
        (16, 500),    # P=8
        (8, 1032),    # P=8: exactly 129 chunks -> group boundary + short group
        (20, 2000),   # P=4, many groups
        (1, 600),     # P=8, degenerate N=1 (every row softmaxes to 1.0)
    ],
)
def test_more_than_128_rows(inst_file, n, rows):
    """Rows beyond 128 used to silently corrupt: lr_row feeds MULT.RC.VE's
    `src` scalar-select and AGG's dest slot, both of which index R0 for 0..127
    and switch to the never-loaded R1 at 128 (see STATUS.md's row-count
    overflow). The kernel now processes groups of at most 128/P chunks (= 128
    logical rows), restarting lr_row each group, so the index can't reach 128.
    These configs all cross at least one group boundary.
    """
    rng = np.random.RandomState(rows)
    x = (rng.randn(rows, n) * 3.0).astype(np.float32)
    out = _run(inst_file, x, n)
    ref = _reference(x)
    assert np.abs(out - ref).max() < 1e-4


CASES = {
    "default": random_case(axis=1, defaults={'n': 32, 'rows': 8, 'seed': 0, 'scale': 5.0}, max_cycles=4000000),
}


def test_default_case():
    run_case("softmax_rows_partial", CASES["default"])
