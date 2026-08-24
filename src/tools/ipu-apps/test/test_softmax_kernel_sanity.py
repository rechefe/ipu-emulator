"""Kernel sanity for softmax_rows_partial / softmax_columns_packed at the
shapes the full-layer chain actually uses (N_TOK in {64, 16}), before wiring
either kernel into test_full_layer_l4.py / test_full_layer_l5.py.

Distinct from test_softmax_kernels_vs_stub.py, which exercises small
diagnostic shapes (rows=1, rows=2) to isolate the rc_idx addressing path.
Here the goal is coverage at production shapes plus a cheap structural
invariant -- every output in [0, 1], every row/column sums to 1 -- that
should have existed from the start: it catches the entire class of
addressing bugs (wrong row landed in the wrong slot, a partition dropped,
a group not visited) for free, independent of any particular reference
value being right.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_as.label import reset_labels

from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp
from ipu_apps.softmax.softmax_columns_packed import SoftmaxColumnsPackedApp

_SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps/softmax"


def _ref_rows(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=1, keepdims=True)
    z = np.exp(x - m)
    return z / z.sum(axis=1, keepdims=True)


def _ref_cols(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=0, keepdims=True)
    z = np.exp(x - m)
    return z / z.sum(axis=0, keepdims=True)


def _assemble(tmp_path: Path, asm_rel: str, tag: str) -> Path:
    """See test_softmax_kernels_vs_stub.py's _assemble for why reset_labels()
    is required here (assembler label-table singleton, no automatic reset)."""
    reset_labels()
    inst_path = tmp_path / f"{tag}.bin"
    asm_path = _SRC / asm_rel
    assemble_to_bin_file(asm_path.read_text(), str(inst_path))
    return inst_path


@pytest.fixture(scope="module")
def rows_partial_inst(tmp_path_factory) -> Path:
    tmp_path = tmp_path_factory.mktemp("softmax_rows_partial_asm")
    return _assemble(tmp_path, "softmax_rows_partial/softmax_rows_partial.asm", "rows_partial")


@pytest.fixture(scope="module")
def columns_packed_inst(tmp_path_factory) -> Path:
    tmp_path = tmp_path_factory.mktemp("softmax_columns_packed_asm")
    return _assemble(tmp_path, "softmax_columns_packed/softmax_columns_packed.asm", "columns_packed")


def _run_rows_partial(inst_path: Path, tmp_path: Path, x: np.ndarray) -> np.ndarray:
    rows, n = x.shape
    inp = tmp_path / "in.bin"
    outp = tmp_path / "out.bin"
    inp.write_bytes(x.astype(np.float32).tobytes())
    app = SoftmaxRowsPartialApp(inst_path=inst_path, input_path=inp, output_path=outp, n=n, rows=rows)
    app.run(max_cycles=20_000_000)
    return np.frombuffer(outp.read_bytes(), dtype=np.float32).reshape(rows, n)


def _run_columns_packed(inst_path: Path, tmp_path: Path, x: np.ndarray) -> np.ndarray:
    rows, width = x.shape
    inp = tmp_path / "in.bin"
    outp = tmp_path / "out.bin"
    inp.write_bytes(x.astype(np.float32).tobytes())
    app = SoftmaxColumnsPackedApp(inst_path=inst_path, input_path=inp, output_path=outp, rows=rows, width=width)
    app.run(max_cycles=20_000_000)
    return np.frombuffer(outp.read_bytes(), dtype=np.float32).reshape(rows, width)


def _assert_structurally_valid(probs: np.ndarray, *, axis: int, tol: float = 1e-5) -> None:
    """Every element in [0, 1]; every row (axis=1) or column (axis=0) sums to
    1. Independent of any reference value -- catches wrong-row/wrong-group
    addressing bugs (a row landed in the wrong slot, a partition skipped, a
    padding lane leaked into the sum) even if this test's own reference
    comparison had a bug of its own.
    """
    assert probs.min() >= -tol, f"softmax output has values below 0: min={probs.min():.3e}"
    assert probs.max() <= 1.0 + tol, f"softmax output has values above 1: max={probs.max():.3e}"
    sums = probs.sum(axis=axis)
    np.testing.assert_allclose(
        sums, np.ones_like(sums), atol=tol,
        err_msg=f"softmax output does not sum to 1 along axis={axis}",
    )


# ---------------------------------------------------------------------------
# softmax_rows_partial at production N_TOK shapes (N_TOK in {64, 16}),
# multi-row (rows > 1 so the cross-partition repack is exercised).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tok", [64, 16])
@pytest.mark.parametrize("rows", [1, 2, 8, 13])
def test_rows_partial_sanity(rows_partial_inst, tmp_path, n_tok, rows) -> None:
    """rows=13 is deliberately not a multiple of P (=128/ps) for either n_tok,
    forcing the last-chunk zero-padding path in _pack_input/teardown."""
    x = (np.random.RandomState(1000 * n_tok + rows).randn(rows, n_tok) * 3).astype(np.float32)
    got = _run_rows_partial(rows_partial_inst, tmp_path, x)

    _assert_structurally_valid(got, axis=1)
    np.testing.assert_allclose(got, _ref_rows(x), atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# softmax_columns_packed at production N_TOK shapes (rows = N_TOK in
# {64, 16}, width <= 64 so the kernel's packed regime always applies).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [64, 16])
@pytest.mark.parametrize("n_tok_rows", [1, 2, 8, 13, 64])
def test_columns_packed_sanity(columns_packed_inst, tmp_path, width, n_tok_rows) -> None:
    """n_tok_rows=64 matches the full N_TOK=64 chain's real row count (one
    row per key, N_TOK keys); n_tok_rows=13 exercises the non-multiple-of-rpv
    tail-padding path."""
    x = (np.random.RandomState(2000 * width + n_tok_rows).randn(n_tok_rows, width) * 3).astype(np.float32)
    got = _run_columns_packed(columns_packed_inst, tmp_path, x)

    _assert_structurally_valid(got, axis=0)
    np.testing.assert_allclose(got, _ref_cols(x), atol=1e-6, rtol=1e-5)
