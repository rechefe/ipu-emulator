"""Self-check tests for the residual_add bootcamp exercise (add -> subtract).

This is the drill described in residual_add/exercise/residual_sub_stub.asm:
the learner edits a copy of residual_add.asm to swap the second ACC.ADD for
ACC.SUB so it computes out[v] = A[v] - B[v] instead of A[v] + B[v].

- test_residual_sub_solution_matches_numpy assembles the answer key
  (residual_sub_solution.asm) and checks it against a numpy A-B reference.
  This is the test a learner should model their own fixed-up stub against.
- test_stub_as_shipped_still_computes_addition documents the starting state:
  the stub ships with the TODO unfilled (ACC.ADD still in place), so it
  still computes A+B, NOT A-B. This is expected -- the stub is meant to be
  edited by hand, not solved by this test suite -- and the assertion here
  confirms *why* the exercise exists (comparing against the solution's
  correct A-B output would fail until the TODO is fixed).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.bootcamp.residual_add import ResidualAddApp, LANES, ROW_BYTES_OUT
from ipu_apps.bootcamp.residual_add.exercise import (
    SOLUTION_ASM_PATH,
    STUB_ASM_PATH,
)


def _run(asm_path: Path, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num_vectors = a.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "prog.bin"
        a_file = tmp / "a.bin"
        b_file = tmp / "b.bin"
        assemble_to_bin_file(asm_path.read_text(), str(inst_file))
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


@pytest.mark.parametrize("num_vectors,seed", [(1, 0), (3, 1), (8, 2)])
def test_residual_sub_solution_matches_numpy(num_vectors, seed):
    """The answer key must compute A[v] - B[v], matching numpy exactly."""
    rng = np.random.RandomState(seed)
    shape = (num_vectors, LANES)
    a = rng.randint(-64, 64, size=shape)
    b = rng.randint(-64, 64, size=shape)
    ref = a.astype(np.int32) - b.astype(np.int32)

    out = _run(SOLUTION_ASM_PATH, a, b)
    assert np.array_equal(out, ref)


def test_stub_as_shipped_still_computes_addition():
    """Documents the exercise's starting state (TODO unfilled).

    The stub ships with ACC.ADD still in place (the fix is left to the
    learner), so as shipped it still computes A[v] + B[v], not A[v] - B[v].
    This test does NOT require the stub to be solved -- it asserts the
    *expected pre-fix behaviour*, i.e. it matches the addition reference and
    diverges from the subtraction reference. Once you've edited your own
    copy to swap in ACC.SUB, re-check it against
    test_residual_sub_solution_matches_numpy's reference instead.
    """
    rng = np.random.RandomState(0)
    shape = (4, LANES)
    a = rng.randint(-64, 64, size=shape)
    b = rng.randint(-64, 64, size=shape)
    add_ref = a.astype(np.int32) + b.astype(np.int32)
    sub_ref = a.astype(np.int32) - b.astype(np.int32)

    out = _run(STUB_ASM_PATH, a, b)
    assert np.array_equal(out, add_ref)
    assert not np.array_equal(out, sub_ref)
