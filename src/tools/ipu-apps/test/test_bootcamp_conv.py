"""Correctness tests for bootcamp drill 2: conv (1D single-channel 3-tap conv).

Assembles conv.asm in-process and compares against a numpy "valid" 1D
convolution reference (positions 1..126; edge lanes 0/127 are not produced,
see conv.asm's "WHY NO MASKING" note) across weights and row counts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.bootcamp.conv import ConvApp, LANES, ROW_BYTES_OUT, VALID_LO, VALID_HI

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/bootcamp/conv/conv.asm"
)


def _numpy_reference(rows: np.ndarray, weights: tuple[int, int, int]) -> np.ndarray:
    w_left, w_center, w_right = weights
    rows64 = rows.astype(np.int64)
    left = rows64[:, VALID_LO - 1 : VALID_HI - 1]
    center = rows64[:, VALID_LO:VALID_HI]
    right = rows64[:, VALID_LO + 1 : VALID_HI + 1]
    return w_left * left + w_center * center + w_right * right


def _run(inst_file: Path, data: np.ndarray, weights: tuple[int, int, int]) -> np.ndarray:
    num_rows = data.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(data.astype(np.int8).tobytes())
        app = ConvApp(
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


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "conv.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


@pytest.mark.parametrize(
    "num_rows,weights,seed",
    [
        (1, (1, 1, 1), 0),
        (2, (1, 2, -1), 1),
        (4, (0, 1, 0), 2),      # identity kernel: out == in (shifted)
        (8, (-1, 0, 1), 3),     # discrete derivative kernel
        (16, (3, -2, 5), 4),
    ],
)
def test_conv_matches_numpy(inst_file, num_rows, weights, seed):
    rng = np.random.RandomState(seed)
    data = rng.randint(-32, 32, size=(num_rows, LANES))
    ref = _numpy_reference(data, weights)

    out = _run(inst_file, data, weights)
    assert np.array_equal(out, ref)


def test_conv_identity_kernel_passes_through(inst_file):
    """Weight (0, 1, 0) must reproduce the center input unchanged."""
    data = np.arange(LANES, dtype=np.int8).reshape(1, LANES) - 64  # spread across INT8 range
    out = _run(inst_file, data, (0, 1, 0))
    expected = data[:, VALID_LO:VALID_HI].astype(np.int64)
    assert np.array_equal(out, expected)


def test_conv_zero_kernel_is_zero(inst_file):
    data = np.full((2, LANES), 5, dtype=np.int8)
    out = _run(inst_file, data, (0, 0, 0))
    assert np.all(out == 0)
