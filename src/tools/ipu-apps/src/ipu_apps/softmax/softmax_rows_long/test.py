"""Reusable cases and numerical regression coverage for softmax_rows_long."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_apps.softmax.test_support import assemble_kernel, run_random_array
from ipu_apps.kernel_registry.cases import run_case
from ipu_apps.softmax.softmax_rows_long.cases import CASES

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


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory) -> Path:
    return assemble_kernel(ASM_PATH, tmp_path_factory)


@pytest.mark.parametrize("rows,n,scale,seed", TEST_CONFIGS)
def test_softmax_rows_long_matches_numpy(inst_file, rows, n, scale, seed):
    cycles, _ = run_random_array("softmax_rows_long", inst_file, rows, n, scale, seed, axis=1)
    assert cycles > 0


def test_default_case(inst_file):
    run_case("softmax_rows_long", CASES["default"], inst_path=inst_file)
