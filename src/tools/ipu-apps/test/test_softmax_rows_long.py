"""Pytest wrapper for the softmax_rows_long suite.

Reuses the standalone runner's reference + config list (next to the app) and
parametrizes them so the configs run as individual pytest cases. The asm is
assembled once per session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_rows_long.test_softmax_rows_long import (
    ASM_PATH,
    TEST_CONFIGS,
    run_one,
)


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
