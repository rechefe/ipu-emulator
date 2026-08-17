"""Self-check tests for the softmax bootcamp exercise (re-add stability fix).

This is the drill described in softmax/exercise/softmax_unstable_stub.asm:
the learner re-adds Pass 1 (the max reduction) and Pass 2's max-subtract
step, which the stub ships with stubbed out -- Pass 2 in the stub feeds raw
c*x[r,j] directly into exp2 instead of c*x[r,j] - maxvec[r].

- test_softmax_solution_matches_numpy assembles the answer key
  (softmax_solution.asm, logically identical to ../softmax.asm) and checks
  it against a numpy softmax reference across row counts and logit
  magnitudes, including a large-magnitude (scale=50) case. This is the test
  a learner should model their own fixed-up stub against.
- test_stub_diverges_on_large_logits documents the stub's failure mode: on
  large-magnitude logits (scale=50), the unstable Pass 2 overflows FP32's
  exp2 range, producing inf/nan and hence output that diverges sharply from
  numpy softmax. This test does NOT require the stub to be solved -- it
  documents the pre-fix behaviour the exercise asks you to change.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.bootcamp.softmax import LANES, ROW_BYTES
from ipu_apps.bootcamp.softmax.exercise import (
    SoftmaxExerciseApp,
    SOLUTION_ASM_PATH,
    STUB_ASM_PATH,
)


def _reference(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def _run(asm_path: Path, x: np.ndarray) -> np.ndarray:
    rows = x.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        input_file = Path(tmp) / "input.bin"
        input_file.write_bytes(x.astype(np.float32).tobytes())
        inst_file = Path(tmp) / "prog.bin"
        assemble_to_bin_file(asm_path.read_text(), str(inst_file))
        app = SoftmaxExerciseApp(
            inst_path=inst_file,
            input_path=input_file,
            output_path=None,
            rows=rows,
        )
        state, _ = app.run(max_cycles=1_000_000)
        raw = state.xmem.read_address(app.output_base, rows * ROW_BYTES)
    return np.frombuffer(raw, dtype=np.float32).reshape(rows, LANES)


@pytest.mark.parametrize(
    "rows,scale,seed",
    [
        (1, 3.0, 0),
        (4, 3.0, 0),
        (8, 5.0, 1),
        (32, 5.0, 2),
        (8, 50.0, 4),    # numerical-stability: large magnitudes
        (8, 0.01, 5),    # near-uniform logits
    ],
)
def test_softmax_solution_matches_numpy(rows, scale, seed):
    rng = np.random.RandomState(seed)
    x = (rng.randn(rows, LANES) * scale).astype(np.float32)
    out = _run(SOLUTION_ASM_PATH, x)
    ref = _reference(x)

    assert np.abs(out - ref).max() < 1e-4
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_stub_diverges_on_large_logits():
    """Documents the exercise's starting state (stability fix unfilled).

    The stub skips the max-subtract, so Pass 2 computes 2^(c*x[r,j])
    directly. For logits with scale=50, c*x[r,j] (c = log2(e) ~= 1.44) is
    large enough that 2^(c*x[r,j]) overflows FP32 for the largest logits in
    a row. In this emulator that overflow surfaces as a hard OverflowError
    from ACTIVATE.QUANTIZE's FP32 pack step (rather than silently producing
    inf) -- confirmed empirically below. Either way (crash, or -- on a
    build where it doesn't raise -- inf/nan propagating into a garbage
    output) this is the "diverges from a numpy softmax reference" failure
    mode the exercise describes. This test does not require the stub to be
    solved -- it documents the pre-fix failure the exercise asks you to fix.
    Once you've re-added Pass 1 and the Pass 2 subtract, check your own copy
    against test_softmax_solution_matches_numpy's reference instead.
    """
    rng = np.random.RandomState(4)
    x = (rng.randn(8, LANES) * 50.0).astype(np.float32)
    ref = _reference(x)

    try:
        out = _run(STUB_ASM_PATH, x)
    except OverflowError:
        # FP32 can't represent 2^(c*x_j) for the largest logits at this
        # scale -- ACTIVATE.QUANTIZE's pack step raises rather than
        # producing inf. This is exactly the numerical-stability failure
        # the max-subtract exists to prevent.
        return

    finite = np.isfinite(out)
    diverges = (not np.all(finite)) or np.abs(out - ref).max() > 1e-2
    assert diverges

    # Meanwhile, small-magnitude logits are unaffected by the missing
    # stability fix (no overflow risk at this scale), so the stub still
    # matches numpy there -- this isolates the failure to the large-|x| case
    # the stability trick exists for.
    rng2 = np.random.RandomState(0)
    x_small = (rng2.randn(4, LANES) * 3.0).astype(np.float32)
    ref_small = _reference(x_small)
    out_small = _run(STUB_ASM_PATH, x_small)
    assert np.abs(out_small - ref_small).max() < 1e-3
