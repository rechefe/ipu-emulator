"""Pytest wrapper for the pointwise_conv_unified_bn_activation suite.

Reuses the standalone runner's PyTorch reference + config list (which live
next to the app, in
``pointwise_conv_unified_bn_activation/test_bn_activation.py``) and
parametrizes them so the configs run as individual pytest cases. The asm is
assembled once per session.

This app runs exclusively on the FP32 wide-vector debug datapath -- there is
no narrow/INT8 mode to separately verify.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified_bn_activation.test_bn_activation import (
    ASM_PATH,
    TEST_CONFIGS,
    run_one,
)

_TOL = 1e-3


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("pw_unified_bn")
    inst_file = tmp / "pointwise_conv_unified_bn_activation.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
    return inst_file


@pytest.mark.parametrize("height,width,in_ch,out_ch", TEST_CONFIGS)
def test_pointwise_conv_unified_bn_activation(
    inst_file: Path, height: int, width: int, in_ch: int, out_ch: int
) -> None:
    cycles, max_diff, actual, expected = run_one(
        inst_file, height, width, in_ch, out_ch
    )
    assert cycles > 0
    assert max_diff < _TOL, (
        f"max diff {max_diff:.3e} for {height}x{width} ic={in_ch} oc={out_ch}\n"
        f"  first OC actual: {actual[0, 0, :8]}\n"
        f"  first OC expect: {expected[0, 0, :8]}"
    )
