"""Correctness tests for bootcamp drill 1: residual_add (INT8, LR + XMEM only).

Assembles residual_add.asm in-process and compares against a numpy INT8+INT8
-> INT32 reference across a range of vector counts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.bootcamp.residual_add import ResidualAddApp, LANES, ROW_BYTES_OUT

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/bootcamp/residual_add/residual_add.asm"
)


def _run(inst_file: Path, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num_vectors = a.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a_file = tmp / "a.bin"
        b_file = tmp / "b.bin"
        a_file.write_bytes(a.astype(np.int8).tobytes())
        b_file.write_bytes(b.astype(np.int8).tobytes())
        app = ResidualAddApp(
            inst_path=inst_file,
            input_a_path=a_file,
            input_b_path=b_file,
            output_path=None,
            num_vectors=num_vectors,
        )
        state, _ = app.run(max_cycles=100_000)
        raw = state.xmem.read_address(app.output_base, num_vectors * ROW_BYTES_OUT)
    return np.frombuffer(raw, dtype="<i4").reshape(num_vectors, LANES)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "residual_add.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


@pytest.mark.parametrize("num_vectors,seed", [(1, 0), (2, 1), (3, 2), (8, 3), (16, 4)])
def test_residual_add_matches_numpy(inst_file, num_vectors, seed):
    rng = np.random.RandomState(seed)
    shape = (num_vectors, LANES)
    a = rng.randint(-64, 64, size=shape)
    b = rng.randint(-64, 64, size=shape)
    ref = a.astype(np.int32) + b.astype(np.int32)

    out = _run(inst_file, a, b)
    assert np.array_equal(out, ref)


def test_residual_add_int8_extremes(inst_file):
    """INT8 range boundaries must add correctly in the INT32 accumulator."""
    a = np.full((1, LANES), 127, dtype=np.int8)
    b = np.full((1, LANES), 127, dtype=np.int8)
    ref = a.astype(np.int32) + b.astype(np.int32)

    out = _run(inst_file, a, b)
    assert np.array_equal(out, ref)


def test_residual_add_negative_values(inst_file):
    a = np.full((2, LANES), -128, dtype=np.int8)
    b = np.full((2, LANES), -1, dtype=np.int8)
    ref = a.astype(np.int32) + b.astype(np.int32)

    out = _run(inst_file, a, b)
    assert np.array_equal(out, ref)
