"""Self-check tests for the conv bootcamp exercise (widen 3-tap -> 5-tap).

This is the drill described in conv/exercise/conv5_stub.asm: the learner
adds two more MULT.RC.VE/ACC.ADD pairs (rc_idx = -2/+2) and two more CR
tap-weight registers to widen conv.asm's 3-tap convolution to 5 taps. The
valid output range shrinks from p=1..126 to p=2..125.

- test_conv5_solution_matches_numpy assembles the answer key
  (conv5_solution.asm) and checks it against a numpy 5-tap valid-conv
  reference. This is the test a learner should model their own filled-in
  stub against.
- test_stub_as_shipped_still_only_3_tap documents the starting state: the
  stub ships with the two outer taps commented out as TODOs, so as shipped
  it still behaves like a 3-tap conv (matching the 3-tap reference on the
  shared central range) rather than the 5-tap reference. This is expected --
  the stub is meant to be edited by hand, not solved by this test suite.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.bootcamp.conv import LANES, ROW_BYTES_OUT
from ipu_apps.bootcamp.conv.exercise import (
    Conv5App,
    SOLUTION_ASM_PATH,
    STUB_ASM_PATH,
    VALID_LO,
    VALID_HI,
)


def _numpy_reference_5tap(rows: np.ndarray, weights: tuple[int, int, int, int, int]) -> np.ndarray:
    w_l2, w_l1, w_c, w_r1, w_r2 = weights
    rows64 = rows.astype(np.int64)
    t_l2 = rows64[:, VALID_LO - 2 : VALID_HI - 2]
    t_l1 = rows64[:, VALID_LO - 1 : VALID_HI - 1]
    t_c = rows64[:, VALID_LO:VALID_HI]
    t_r1 = rows64[:, VALID_LO + 1 : VALID_HI + 1]
    t_r2 = rows64[:, VALID_LO + 2 : VALID_HI + 2]
    return w_l2 * t_l2 + w_l1 * t_l1 + w_c * t_c + w_r1 * t_r1 + w_r2 * t_r2


def _run(asm_path: Path, data: np.ndarray, weights: tuple[int, int, int, int, int]) -> np.ndarray:
    num_rows = data.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "prog.bin"
        input_file = tmp / "input.bin"
        assemble_to_bin_file(asm_path.read_text(), str(inst_file))
        input_file.write_bytes(data.astype(np.int8).tobytes())
        app = Conv5App(
            inst_path=inst_file,
            input_path=input_file,
            output_path=None,
            num_rows=num_rows,
            weights=weights,
        )
        state, _ = app.run(max_cycles=100_000)
        raw = state.xmem.read_address(app.output_base, num_rows * ROW_BYTES_OUT)
    out = np.frombuffer(raw, dtype="<i4").reshape(num_rows, LANES)
    return out[:, VALID_LO:VALID_HI]


@pytest.mark.parametrize(
    "num_rows,weights,seed",
    [
        (1, (1, 1, 1, 1, 1), 0),
        (2, (1, 2, 3, 2, 1), 1),
        (4, (0, 0, 1, 0, 0), 2),   # identity kernel: out == in (shifted)
        (8, (-1, 0, 0, 0, 1), 3),  # wide discrete derivative
        (16, (1, -2, 3, -2, 1), 4),
    ],
)
def test_conv5_solution_matches_numpy(num_rows, weights, seed):
    rng = np.random.RandomState(seed)
    data = rng.randint(-16, 16, size=(num_rows, LANES))
    ref = _numpy_reference_5tap(data, weights)

    out = _run(SOLUTION_ASM_PATH, data, weights)
    assert np.array_equal(out, ref)


def test_conv5_zero_kernel_is_zero():
    data = np.full((2, LANES), 5, dtype=np.int8)
    out = _run(SOLUTION_ASM_PATH, data, (0, 0, 0, 0, 0))
    assert np.all(out == 0)


def test_stub_as_shipped_still_only_3_tap():
    """Documents the exercise's starting state (outer taps unfilled).

    The stub ships with the outer taps (rc_idx=-2/+2) commented out, so as
    shipped it still behaves like a 3-tap conv: on the shared central range
    it matches a 3-tap reference using the stub's middle three weights, and
    it does NOT match the full 5-tap reference. This test does not require
    the stub to be solved -- it documents the pre-fix behaviour the
    exercise asks you to change. Once you've filled in the two outer taps,
    check your own copy against test_conv5_solution_matches_numpy's
    reference instead.
    """
    rng = np.random.RandomState(0)
    num_rows = 4
    weights = (9, 1, 2, -1, 9)  # outer weights (9) chosen to be obviously wrong if applied
    data = rng.randint(-16, 16, size=(num_rows, LANES))

    out = _run(STUB_ASM_PATH, data, weights)

    full_5tap_ref = _numpy_reference_5tap(data, weights)
    assert not np.array_equal(out, full_5tap_ref)

    # Stub still computes a 3-tap conv using the middle weights (w[-1], w[0], w[1]),
    # just over the narrower 5-tap valid range.
    w_l2, w_l1, w_c, w_r1, w_r2 = weights
    rows64 = data.astype(np.int64)
    t_l1 = rows64[:, VALID_LO - 1 : VALID_HI - 1]
    t_c = rows64[:, VALID_LO:VALID_HI]
    t_r1 = rows64[:, VALID_LO + 1 : VALID_HI + 1]
    three_tap_ref = w_l1 * t_l1 + w_c * t_c + w_r1 * t_r1
    assert np.array_equal(out, three_tap_ref)
