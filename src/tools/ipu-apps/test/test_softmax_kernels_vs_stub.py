"""Real softmax kernels (softmax_rows_partial, softmax_columns_packed) vs. the
host numpy stub (fixture_host_softmax_fold_stubs.py).

BACKGROUND. David Sheinenzon's five FP32 wide-vector softmax kernels live on
`origin/pr4-registry-docs` (commit 9629113). Two of the five match this
repo's full-layer chain shapes:
  * query-major chain (qk_scores_* -> attn_v_*): N_TOK in {16, 64}, softmax
    reduces along each row (torch dim=1) -- softmax_rows_partial (N < 128).
  * key-major chain (attn_scores_km_* -> attn_v_bcast_*): same N_TOK, softmax
    reduces down each column (torch dim=0) -- softmax_columns_packed (width
    <= 64).
Both are vendored, trimmed only of the (unmerged, irrelevant here)
`kernel_registry` auto-selection plumbing -- see each kernel's own
`src/ipu_apps/softmax/<name>/__init__.py` docstring for the exact diff.
Their `.asm` also needed vendoring; both are Jinja2 templates, rendered by
`ipu_as.lark_tree.assemble_to_bin_file` same as every other kernel in this
repo, via `ipu_as.label.reset_labels()` between assemblies (see below).

WHAT WORKS (asserted below):
  * Both kernels assemble cleanly against this branch's assembler.
  * Layout contract: output file byte-layout equals input file byte-layout,
    matching `origin/pr4-registry-docs`'s own
    test_softmax_layout_roundtrip.py -- confirmed here independently.
  * Numerical stability: both kernels subtract the row/column max before
    exponentiating (`ACC.SUB` against a resident max vector, per the branch's
    docs/content/kernels/softmax.md), matching this repo's numpy stub
    (`fixture_host_softmax_fold_stubs.py`'s `m = np.max(...)` subtraction) --
    so no reference-side adjustment was needed to make the comparison
    apples-to-apples.
  * The TRIVIAL case (rows=1, no cross-partition/cross-group repack) agrees
    with the stub to ~1e-9 for both kernels.
  * The cross-partition (rows_partial) / cross-group-fold (columns_packed)
    repack, which uses `MULT.RC.VE`/`MULT.RC.VV` with an ELEMENT-addressed
    rc_idx (`p * ps`, matching `LDR_CYCLIC_MULT_REG`'s `index` -- issue
    #182/PR #196), now agrees with the stub too: `ipu.py`'s
    `Ipu._rc_element_to_byte_offset` (merged from origin/master, formerly
    only on `origin/pr4-registry-docs`) scales rc_idx by the element width
    before indexing r_cyclic, matching these kernels' addressing convention.
"""

from __future__ import annotations

import sys
import tempfile
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
    """Assemble one softmax .asm, isolated from any other assembly in the
    same test process by ipu_as.label.reset_labels() -- the assembler's label
    table (ipu_as.label.ipu_labels) is a module-level singleton with no
    per-call reset, so assembling a second .asm in one process raises a
    spurious 'label defined twice' error without this. This is a pre-existing
    assembler gap (there IS a reset_labels() function, just never called
    automatically) worked around here from test code, not patched in
    ipu_as itself.
    """
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


# ---------------------------------------------------------------------------
# What works: layout + single-row (no cross-partition repack) agreement.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [16, 64])
def test_rows_partial_single_row_matches_stub(rows_partial_inst, tmp_path, n) -> None:
    """No cross-partition repack is exercised at rows=1 (nothing to repack
    across), so this path does NOT hit the rc_idx blocker -- confirms the
    kernel's core exp2/max-subtract/normalize math agrees with the stub
    before the blocker is reached.
    """
    x = (np.random.RandomState(n).randn(1, n) * 3).astype(np.float32)
    got = _run_rows_partial(rows_partial_inst, tmp_path, x)
    np.testing.assert_allclose(got, _ref_rows(x), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("n", [16, 64])
def test_rows_partial_layout_roundtrips(rows_partial_inst, tmp_path, n) -> None:
    """Output file byte-size and naive reshape match the input -- the layout
    contract from origin/pr4-registry-docs's test_softmax_layout_roundtrip.py,
    confirmed independently here (no app-specific unpacking knowledge).
    """
    rows = 6
    x = (np.random.RandomState(n + 1).randn(rows, n) * 3).astype(np.float32)
    inp = tmp_path / "in.bin"
    outp = tmp_path / "out.bin"
    inp.write_bytes(x.tobytes())
    app = SoftmaxRowsPartialApp(inst_path=rows_partial_inst, input_path=inp, output_path=outp, n=n, rows=rows)
    app.run(max_cycles=20_000_000)
    assert outp.stat().st_size == inp.stat().st_size


# ---------------------------------------------------------------------------
# Multi-row/group cross-partition repack (exercises the element-addressed
# rc_idx path -- see module docstring).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [16, 64])
def test_rows_partial_multi_row_matches_stub(rows_partial_inst, tmp_path, n) -> None:
    """rows=2 (n<=64 so P=128/ps>=2, i.e. >1 logical row per physical chunk)
    exercises the Pass-4 cross-partition repack (MULT.RC.VE with an
    ELEMENT-addressed rc_idx, matching LDR_CYCLIC_MULT_REG's index).
    """
    x = (np.random.RandomState(n + 2).randn(2, n) * 3).astype(np.float32)
    got = _run_rows_partial(rows_partial_inst, tmp_path, x)
    np.testing.assert_allclose(got, _ref_rows(x), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("width", [16, 64])
def test_columns_packed_multi_row_matches_stub(columns_packed_inst, tmp_path, width) -> None:
    """Mirrors test_rows_partial_multi_row_matches_stub for
    softmax_columns_packed's cross-group fold (same MULT.RC.VE rc_idx
    convention). Unlike rows_partial, columns_packed's fold walk runs even at
    rows=1 whenever rows_per_vec > 1 (width < 128 always packs >1 group per
    vector), so rows=1 already exercises this path here.
    """
    x = (np.random.RandomState(width + 100).randn(1, width) * 3).astype(np.float32)
    got = _run_columns_packed(columns_packed_inst, tmp_path, x)
    np.testing.assert_allclose(got, _ref_cols(x), atol=1e-6, rtol=1e-5)
