"""Pytest wrapper for the pointwise_conv_unified correctness suite.

Reuses the standalone runner's reference + config list (which live next to the
app, in ``pointwise_conv_unified/test_unified.py``) and parametrizes them so the
configs run as individual pytest cases. The asm is assembled once per session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified.test_unified import (
    ASM_PATH,
    TEST_CONFIGS,
    run_one,
)


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("pw_unified")
    inst_file = tmp / "pointwise_conv_unified.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
    return inst_file


@pytest.mark.parametrize("rows,cols,in_ch,out_ch", TEST_CONFIGS)
def test_pointwise_conv_unified(
    inst_file: Path, rows: int, cols: int, in_ch: int, out_ch: int
) -> None:
    cycles, mismatches, actual, expected = run_one(
        inst_file, rows, cols, in_ch, out_ch
    )
    assert cycles > 0
    assert mismatches == 0, (
        f"{mismatches} mismatches for {rows}x{cols} ic={in_ch} oc={out_ch}\n"
        f"  first OC actual: {actual[0, 0, :8]}\n"
        f"  first OC expect: {expected[0, 0, :8]}"
    )
