"""Correctness tests for the row-softmax app (FP32 wide-vector).

Assembles softmax_rows.asm in-process each run and compares the output against
a numpy softmax reference across a range of row counts and logit magnitudes,
including numerical-stability (large |x|) and degenerate (constant row) cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_rows import (
    SoftmaxRowsApp,
    LANES,
    ROW_BYTES,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/softmax/softmax_rows/softmax_rows.asm"
)


def _reference(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def _run(inst_file: Path, x: np.ndarray) -> np.ndarray:
    rows = x.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        input_file = Path(tmp) / "input.bin"
        input_file.write_bytes(x.astype(np.float32).tobytes())
        app = SoftmaxRowsApp(
            inst_path=inst_file,
            input_path=input_file,
            output_path=None,
            rows=rows,
        )
        state, _ = app.run(max_cycles=8_000_000)
        # Output region is contiguous from app.output_base; for K>128 it spans
        # several groups but the rows stay row-major and contiguous.
        raw = state.xmem.read_address(app.output_base, rows * ROW_BYTES)
    return np.frombuffer(raw, dtype=np.float32).reshape(rows, LANES)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "softmax_rows.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


@pytest.mark.parametrize(
    "rows,scale,seed",
    [
        (1, 3.0, 0),
        (4, 3.0, 0),
        (8, 5.0, 1),
        (32, 5.0, 2),
        (128, 5.0, 3),   # one full group (the original target case)
        (8, 50.0, 4),    # numerical-stability: large magnitudes
        (8, 0.01, 5),    # near-uniform logits
        (129, 3.0, 6),   # K>128: 2 groups, last group 1 real row (rest padded)
        (200, 4.0, 7),   # 2 groups, partial last group
        (256, 3.0, 8),   # exactly 2 full groups
        (384, 5.0, 9),   # 3 full groups
        (500, 3.0, 10),  # 4 groups, partial last group
    ],
)
def test_softmax_matches_numpy(inst_file, rows, scale, seed):
    rng = np.random.RandomState(seed)
    x = (rng.randn(rows, LANES) * scale).astype(np.float32)
    out = _run(inst_file, x)
    ref = _reference(x)

    assert np.abs(out - ref).max() < 1e-4
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_constant_row_is_uniform(inst_file):
    """A row of identical logits must produce the uniform distribution."""
    x = np.full((2, LANES), 3.7, dtype=np.float32)
    out = _run(inst_file, x)
    assert np.allclose(out, 1.0 / LANES, atol=1e-6)


def test_argmax_preserved(inst_file):
    """The largest logit must map to the largest probability per row."""
    rng = np.random.RandomState(7)
    x = (rng.randn(16, LANES) * 4.0).astype(np.float32)
    out = _run(inst_file, x)
    assert np.array_equal(out.argmax(axis=1), x.argmax(axis=1))
