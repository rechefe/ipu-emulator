"""Reusable cases and numerical regression coverage for softmax_rows_long."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.softmax.test_support import random_case, reference, run_array
from ipu_apps.kernel_registry.cases import run_case

ASM_PATH = Path(__file__).with_name("softmax_rows_long.asm")

TEST_CONFIGS = [
    (1, 129, 3.0, 0),     # smallest long row: 1 full chunk + 1 tail
    (4, 200, 4.0, 1),     # 1 full + 72 tail
    (8, 256 + 1, 3.0, 2), # 2 full + 1 tail
    (8, 300, 5.0, 3),     # 2 full + 44 tail
    (16, 130, 50.0, 4),   # numerical stability (large |x|)
    (8, 300, 0.01, 5),    # near-uniform
    (32, 384 + 17, 3.0, 6),  # 3 full + 17 tail
    (128, 129, 4.0, 7),   # max rows in one group
    # n % 128 == 0: exactly full_chunks whole chunks, NO tail chunk. The tail
    # block still executes with valid_elements=0 (CR8), which makes its running
    # AGG.MAX/AGG.SUM exact no-ops -- so the same kernel covers this shape.
    (6, 256, 5.0, 8),     # 2 full chunks, no tail
    (6, 384, 5.0, 9),     # 3 full chunks, no tail
    (4, 512, 3.0, 10),    # 4 full chunks, no tail
    (2, 1024, 4.0, 11),   # 8 full chunks, no tail
    # >128 rows: maxvec/rvec hold one slot per row in a single 128-element vector,
    # so the kernel runs groups of <=128 rows (all four passes per group). Row
    # indices restart each group, which is what keeps them out of the R1 range
    # that MULT.RC.VE's `src` would otherwise select. See the .asm group loop.
    (129, 129, 3.0, 12),  # one row past a full group
    (200, 200, 4.0, 13),
    (256, 130, 3.0, 14),  # exactly two full groups
    (300, 129, 5.0, 15),  # two full groups + a short one
]



def run_one(inst_file, rows, n, scale, seed):
    x = (np.random.RandomState(seed).randn(rows, n) * scale).astype(np.float32)
    cycles, out = run_array("softmax_rows_long", inst_file, x, 1)
    ref = reference(x, 1)
    return cycles, float(np.abs(out - ref).max()), out.sum(axis=1), out, ref


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("softmax_rows_long")
    inst = tmp / "softmax_rows_long.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst))
    return inst


@pytest.mark.parametrize("rows,n,scale,seed", TEST_CONFIGS)
def test_softmax_rows_long_matches_numpy(inst_file, rows, n, scale, seed):
    cycles, max_abs, sums, out, ref = run_one(inst_file, rows, n, scale, seed)
    assert cycles > 0
    assert max_abs < 1e-4, (
        f"max abs error {max_abs:.2e} for rows={rows} n={n} scale={scale}\n"
        f"  row0 out[:6]: {out[0, :6]}\n"
        f"  row0 ref[:6]: {ref[0, :6]}"
    )
    assert np.allclose(sums, 1.0, atol=1e-5), f"row sums not 1.0: {sums[:8]}"


CASES = {
    "default": random_case(axis=1, defaults={'rows': 8, 'n': 300, 'scale': 5.0, 'seed': 0}, max_cycles=8000000),
}


def test_default_case():
    run_case("softmax_rows_long", CASES["default"])
