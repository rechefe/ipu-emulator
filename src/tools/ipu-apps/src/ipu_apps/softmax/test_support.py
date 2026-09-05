"""Shared row-major softmax cases; kernel-specific packing stays in harnesses."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase, run_case


def reference(x, axis):
    exp = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


def prepare_array(workspace, x, axis):
    inp, out = workspace / "input.bin", workspace / "output.bin"
    inp.write_bytes(x.astype(np.float32).tobytes())
    expected = reference(x, axis)

    def check():
        actual = np.frombuffer(out.read_bytes(), dtype=np.float32).reshape(x.shape)
        assert np.abs(actual - expected).max() < 1e-4
        assert np.allclose(actual.sum(axis=axis), 1.0, atol=1e-5)

    return PreparedCase({"shape": x.shape, "dim": axis},
                        {"input_path": inp, "output_path": out}, check)


def random_case(*, axis, defaults, max_cycles):
    def prepare(workspace, **options):
        rows = options["rows"]
        width = options.get("n", options.get("width", 128))
        x = (np.random.RandomState(options["seed"]).randn(rows, width)
             * options["scale"]).astype(np.float32)
        return prepare_array(workspace, x, axis)

    return KernelCase(prepare, defaults, max_cycles)


def run_array(kernel, inst_file, x, axis, max_cycles=8_000_000):
    case = KernelCase(lambda workspace: prepare_array(workspace, x, axis), max_cycles=max_cycles)
    with TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        _, cycles = run_case(kernel, case, workspace=workspace, inst_path=inst_file)
        out = np.frombuffer((workspace / "output.bin").read_bytes(), dtype=np.float32).reshape(x.shape)
    return cycles, out
