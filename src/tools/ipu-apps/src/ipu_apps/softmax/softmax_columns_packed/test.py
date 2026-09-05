"""Reusable cases and numerical regression coverage for softmax_columns_packed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.softmax.test_support import random_case, reference, run_array
from ipu_apps.kernel_registry.cases import run_case

ASM_PATH = Path(__file__).with_name("softmax_columns_packed.asm")

TEST_CONFIGS = [
    (16, 64, 3.0, 0),     # 2 rows/vec, clean width
    (64, 32, 4.0, 1),     # 4 rows/vec
    (100, 16, 3.0, 2),    # 8 rows/vec, rows not a multiple of rpv (tail)
    (33, 16, 5.0, 3),     # 8 rows/vec, 33 rows -> 5 vectors (tail of 1)
    (16, 33, 3.0, 4),     # width 33 -> pad to 64 (intra-group padding)
    (32, 20, 4.0, 5),     # width 20 -> pad to 32
    (16, 15, 50.0, 6),    # width 15 -> pad to 16, large |x| (stability)
    (8, 10, 0.01, 7),     # width 10 -> pad to 16, near-uniform
    (1, 64, 3.0, 8),      # single row -> softmax all 1.0
]



def run_one(inst_file, rows, width, scale, seed):
    x = (np.random.RandomState(seed).randn(rows, width) * scale).astype(np.float32)
    cycles, out = run_array("softmax_columns_packed", inst_file, x, 0)
    ref = reference(x, 0)
    return cycles, float(np.abs(out - ref).max()), out.sum(axis=0), out, ref


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("softmax_columns_packed")
    inst = tmp / "softmax_columns_packed.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst))
    return inst


@pytest.mark.parametrize("rows,width,scale,seed", TEST_CONFIGS)
def test_softmax_columns_packed_matches_numpy(inst_file, rows, width, scale, seed):
    cycles, max_abs, sums, out, ref = run_one(inst_file, rows, width, scale, seed)
    assert cycles > 0
    assert max_abs < 1e-4, (
        f"max abs error {max_abs:.2e} for rows={rows} width={width} scale={scale}\n"
        f"  col0 out[:6]: {out[:6, 0]}\n"
        f"  col0 ref[:6]: {ref[:6, 0]}"
    )
    assert np.allclose(sums, 1.0, atol=1e-5), f"column sums not 1.0: {sums[:8]}"


CASES = {
    "default": random_case(axis=0, defaults={'rows': 64, 'width': 16, 'scale': 3.0, 'seed': 0}, max_cycles=8000000),
}


def test_default_case():
    run_case("softmax_columns_packed", CASES["default"])
