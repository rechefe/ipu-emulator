"""Pytest wrapper for the softmax_columns_packed suite.

Reuses the standalone runner's reference + config list (next to the app) and
parametrizes them so the configs run as individual pytest cases. The asm is
assembled once per session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_columns_packed.test_softmax_columns_packed import (
    ASM_PATH,
    TEST_CONFIGS,
    run_one,
)


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
