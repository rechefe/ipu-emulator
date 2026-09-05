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
        raw = out.read_bytes()
        if len(raw) != expected.size * 4:
            raise ValueError(f"softmax output size mismatch: got {len(raw)} bytes, expected {expected.size * 4}")
        actual = np.frombuffer(raw, dtype=np.float32).reshape(x.shape)
        max_error = float(np.abs(actual - expected).max())
        if not max_error < 1e-4:
            raise ValueError(f"softmax max absolute error {max_error:.6g} exceeds tolerance 0.0001 (shape={x.shape}, axis={axis})")
        sums = actual.sum(axis=axis)
        if not np.allclose(sums, 1.0, atol=1e-5):
            raise ValueError(f"softmax sums differ from 1: max deviation {float(np.abs(sums - 1).max()):.6g}")

    return PreparedCase({"shape": x.shape, "dim": axis},
                        {"input_path": inp, "output_path": out}, check)


def random_case(*, axis, defaults, max_cycles, width_option="width", fixed_width=None):
    if fixed_width is None and width_option not in defaults:
        raise ValueError(f"random case requires a {width_option!r} default")
    def prepare(workspace, **options):
        rows = options["rows"]
        width = fixed_width if fixed_width is not None else options[width_option]
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


def assemble_kernel(asm_path, tmp_path_factory):
    """Assemble once from a module-scoped test fixture."""
    from ipu_as.lark_tree import assemble_to_bin_file

    inst = tmp_path_factory.mktemp(asm_path.stem) / (asm_path.stem + ".bin")
    assemble_to_bin_file(asm_path.read_text(), str(inst))
    return inst


def run_random_array(kernel, inst_file, rows, width, scale, seed, *, axis):
    x = (np.random.RandomState(seed).randn(rows, width) * scale).astype(np.float32)
    return run_array(kernel, inst_file, x, axis)
