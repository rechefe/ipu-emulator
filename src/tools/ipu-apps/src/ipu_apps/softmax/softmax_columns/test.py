"""Reusable cases and numerical regression coverage for softmax_columns."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_apps.softmax.test_support import assemble_kernel, run_random_array
from ipu_apps.kernel_registry.cases import run_case
from ipu_apps.softmax.softmax_columns.cases import CASES

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


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory) -> Path:
    return assemble_kernel(ASM_PATH, tmp_path_factory)


@pytest.mark.parametrize("rows,width,scale,seed", TEST_CONFIGS)
def test_softmax_columns_matches_numpy(inst_file, rows, width, scale, seed):
    cycles, _ = run_random_array("softmax_columns", inst_file, rows, width, scale, seed, axis=0)
    assert cycles > 0


def test_default_case(inst_file):
    run_case("softmax_columns", CASES["default"], inst_path=inst_file)
