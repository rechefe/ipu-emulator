"""Reference loading (E06-09): the four forms, located errors, and evaluate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.params import PlanError
from ipu_harness.reference import evaluate, load_reference

TOML = """\
[kernel]
op = "layernorm"
formats = ["fp32"]

[params]
rows = { type = "int", default = 2 }

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"

[cases.default]
"""

FUNCTIONS = (
    "def reference(x, rows):\n    return {'y': x}\n\n\ndef other(x, rows):\n    return {'y': x}\n"
)


def kernel(tmp_path: Path, golden: str = "", reference: str | None = None) -> KernelManifest:
    directory = tmp_path / "norm" / "layernorm"
    directory.mkdir(parents=True)
    (directory / "layernorm.asm").write_text("HALT ;;\n")
    if reference is not None:
        (directory / "reference.py").write_text(reference)
    (directory / "kernel.toml").write_text(TOML + golden)
    return load_manifest(directory)


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """A shared-reference module ``fake_refs.ops`` with ``layernorm`` on the import path."""
    package = tmp_path / "site" / "fake_refs"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "ops.py").write_text(
        "def layernorm(x, rows):\n    return {'y': -x}\n\n\nnot_a_function = 3\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    return "fake_refs.ops"


def test_an_explicit_function_in_the_kernel_folder(tmp_path):
    fn = load_reference(kernel(tmp_path, '[golden]\nreference = "reference.py:other"\n', FUNCTIONS))
    assert fn is not None and fn.__name__ == "other"


def test_a_shared_reference_by_op(tmp_path, shared):
    fn = load_reference(kernel(tmp_path, '[golden]\nop = "layernorm"\n'), shared=shared)
    assert fn is not None and fn.__name__ == "layernorm" and fn.__module__ == "fake_refs.ops"


def test_an_implicit_reference_py(tmp_path):
    fn = load_reference(kernel(tmp_path, reference=FUNCTIONS))
    assert fn is not None and fn.__name__ == "reference"


def test_no_reference(tmp_path):
    assert load_reference(kernel(tmp_path)) is None


@pytest.mark.parametrize(
    "golden, reference, key, message",
    [
        ('reference = "reference.py:missing"', FUNCTIONS, "golden.reference", "is not a function"),
        (
            'reference = "reference.py:reference"',
            "def reference(:\n",
            "golden.reference",
            "SyntaxError",
        ),
        ('op = "softmax"', None, "golden.op", "fake_refs.ops.softmax is not a function"),
        ('op = "not_a_function"', None, "golden.op", "is not a function"),
    ],
)
def test_a_reference_that_does_not_load_is_located(
    tmp_path, shared, golden, reference, key, message
):
    m = kernel(tmp_path, f"[golden]\n{golden}\n", reference)
    with pytest.raises(PlanError, match=message) as error:
        load_reference(m, shared=shared)
    assert error.value.key == key


def test_a_missing_shared_module_is_located(tmp_path):
    m = kernel(tmp_path, '[golden]\nop = "layernorm"\n')
    with pytest.raises(PlanError, match="no_such_refs.ops is not installed") as error:
        load_reference(m, shared="no_such_refs.ops")
    assert error.value.key == "golden.op"


def test_evaluate_passes_float64_tensors_and_returns_float64_arrays(tmp_path):
    torch = pytest.importorskip("torch")  # test-only; skipped where it is not installed
    m = kernel(tmp_path, reference="def reference(x, rows):\n    return {'y': x * rows}\n")
    fn = load_reference(m)
    assert fn is not None
    out = evaluate(fn, {"x": np.ones((2, 128), dtype=np.float32)}, {"rows": 2})
    assert out["y"].dtype == np.float64 and np.array_equal(out["y"], np.full((2, 128), 2.0))
    assert isinstance(torch.zeros(1), torch.Tensor)
