"""Reusable cases and numerical regression coverage for softmax_columns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.softmax.test_support import random_case, reference, run_array
from ipu_apps.kernel_registry.cases import run_case

ASM_PATH = Path(__file__).with_name("softmax_columns.asm")

TEST_CONFIGS = [
    (8, 128, 3.0, 0),      # single full chunk, few rows
    (64, 128, 4.0, 1),     # square 64x128
    (1, 128, 5.0, 2),      # single row -> softmax is all 1.0
    (16, 130, 3.0, 3),     # width 130 -> padded to 256 (2 chunks)
    (32, 200, 4.0, 4),     # width 200 -> padded to 256
    (128, 256, 3.0, 5),    # full 2-chunk width, many rows
    (64, 192, 50.0, 6),    # large |x| (stability), 192 -> 256
    (10, 129, 0.01, 7),    # near-uniform, 129 -> 256
    (256, 256, 3.0, 8),    # 256 rows (no row-group cap)
    (32, 300, 4.0, 9),     # width 300 -> padded to 384 (3 chunks)
    (16, 460, 3.0, 10),    # width 460 -> padded to 512 (4 chunks)
    (8, 384, 50.0, 11),    # exact 3-chunk width, large |x|
    # Sub-128 widths (65..127): one chunk, mostly padding. Correct because each
    # element is an INDEPENDENT column -- padding elements are their own (all-zero)
    # columns and never enter a real column's reduce. Widths <= 64 belong to
    # softmax_columns_packed, which fits several whole rows per vector.
    (32, 65, 4.0, 12),     # narrowest supported width
    (32, 96, 3.0, 13),
    (32, 127, 5.0, 14),    # widest sub-128 width
    (16, 100, 50.0, 15),   # sub-128 + large |x| (stability)
]



def run_one(inst_file, rows, width, scale, seed):
    x = (np.random.RandomState(seed).randn(rows, width) * scale).astype(np.float32)
    cycles, out = run_array("softmax_columns", inst_file, x, 0)
    ref = reference(x, 0)
    return cycles, float(np.abs(out - ref).max()), out.sum(axis=0), out, ref


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("softmax_columns")
    inst = tmp / "softmax_columns.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst))
    return inst


@pytest.mark.parametrize("rows,width,scale,seed", TEST_CONFIGS)
def test_softmax_columns_matches_numpy(inst_file, rows, width, scale, seed):
    cycles, max_abs, sums, out, ref = run_one(inst_file, rows, width, scale, seed)
    assert cycles > 0
    assert max_abs < 1e-4, (
        f"max abs error {max_abs:.2e} for rows={rows} width={width} scale={scale}\n"
        f"  col0 out[:6]: {out[:6, 0]}\n"
        f"  col0 ref[:6]: {ref[:6, 0]}"
    )
    assert np.allclose(sums, 1.0, atol=1e-5), f"column sums not 1.0: {sums[:8]}"


CASES = {
    "default": random_case(axis=0, defaults={'rows': 64, 'width': 128, 'scale': 5.0, 'seed': 0}, max_cycles=8000000),
}


def test_default_case():
    run_case("softmax_columns", CASES["default"])
