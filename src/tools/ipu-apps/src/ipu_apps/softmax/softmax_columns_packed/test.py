"""Reusable cases and numerical regression coverage for softmax_columns_packed."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_apps.softmax.test_support import assemble_kernel, run_random_array
from ipu_apps.kernel_registry.cases import run_case
from ipu_apps.softmax.softmax_columns_packed.cases import CASES

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


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory) -> Path:
    return assemble_kernel(ASM_PATH, tmp_path_factory)


@pytest.mark.parametrize("rows,width,scale,seed", TEST_CONFIGS)
def test_softmax_columns_packed_matches_numpy(inst_file, rows, width, scale, seed):
    cycles, _ = run_random_array("softmax_columns_packed", inst_file, rows, width, scale, seed, axis=0)
    assert cycles > 0


def test_default_case(inst_file):
    run_case("softmax_columns_packed", CASES["default"], inst_path=inst_file)
