"""Exact input sizes (E07-09, L15): an input larger than its declared shape is
an error, never silently truncated as the old ``matmul_*_x128`` harnesses
did (``raw[:expected]``). The error names the tensor and both shapes."""

from __future__ import annotations

import numpy as np
import pytest

from ipu_emu.errors import ConfigurationError
from ipu_harness import cli
from ipu_harness.cases import expand
from ipu_harness.registry import Registry, kernel_tree
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.formats import ElementFormat

NAME = "matmul_144x288_x128"


@pytest.fixture(scope="module")
def kernel():
    return Registry.load(kernel_tree(), families=["matmul"]).kernel(NAME)


@pytest.mark.parametrize(
    ("tensor", "shape", "declared"),
    [("b", (2, 289, 128), (2, 288, 128)), ("a", (145, 288), (144, 288))],
)
def test_an_oversize_input_is_refused_with_both_shapes(kernel, tmp_path, tensor, shape, declared):
    path = tmp_path / f"{tensor}.npy"
    np.save(path, np.zeros(shape, dtype=np.float32))
    case = expand(kernel)[0]
    with pytest.raises(ConfigurationError) as error:
        run_kernel(kernel, case, ElementFormat.FP32, RunOptions(inputs={tensor: path}))
    assert f"{tensor} has shape {shape}, the case's is {declared}" in str(error.value)


def test_the_command_line_refuses_it_too(tmp_path, capsys):
    """``bazel run //kernels/matmul:matmul_144x288_x128 -- --input b=...``."""
    path = tmp_path / "b.npy"
    np.save(path, np.zeros((2, 289, 128), dtype=np.float32))
    entry = f"{kernel_tree() / 'matmul' / 'registry.toml'}::{NAME}"
    assert cli.main([entry, "--input", f"b={path}"]) == 1
    err = capsys.readouterr().err
    assert "b has shape (2, 289, 128), the case's is (2, 288, 128)" in err
