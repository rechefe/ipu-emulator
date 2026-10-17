"""Cases: sweeps, seeded generators, user data and its priority (E06-08)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.errors import ConfigurationError
from ipu_harness.cases import expand, inputs_for, rng
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.params import PlanError, plan_case
from ipu_isa.formats import ElementFormat as F

SCALE = """\
[kernel]
op = "scale"
formats = ["fp32", "int8", "e4"]

[params]
rows = {{ type = "int", min = 1, default = 4 }}
s    = {{ type = "float" }}

[inputs.x]
shape = ["rows", 128]
{x_extra}
[outputs.y]
shape = ["rows", 128]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[cases.default]
rows = 4
s = 2.0

[cases.one_row]
rows = 1
s = -3.0

[cases.sweep]
rows = [1, 7, 33, 128]
s = 1.5
formats = ["fp32", "e4"]

[cases.mine]
rows = 2
s = 2.0
inputs = {{ x = "data/x.npy" }}
expected = {{ y = "data/y.npy" }}
{extra_cases}"""

#: E4 values next to 0.3: 0.28125 and 0.3125 (3 mantissa bits); 0.3 rounds to 0.3125.
E4_NEAR_0_3 = 0.3125


def scale(tmp_path: Path, *, x_extra: str = "", extra_cases: str = "") -> KernelManifest:
    directory = tmp_path / "scale"
    (directory / "data").mkdir(parents=True)
    (directory / "scale.asm").write_text("HALT ;;\n")
    x = np.arange(-4, 4, dtype=np.float32).repeat(32).reshape(2, 128)
    np.save(directory / "data" / "x.npy", x)
    np.save(directory / "data" / "y.npy", np.zeros((2, 128), dtype=np.float32))
    (directory / "kernel.toml").write_text(SCALE.format(x_extra=x_extra, extra_cases=extra_cases))
    return load_manifest(directory)


def planned(manifest: KernelManifest, name: str, fmt: F = F.FP32):
    case = next(c for c in expand(manifest) if c.name == name)
    return case, plan_case(manifest, case.values, fmt, case=case.decl)


def generator(manifest: KernelManifest, body: str) -> None:
    (manifest.directory / "gen.py").write_text(
        f"import numpy as np\n\n\ndef make(plan, rng):\n    {body}\n"
    )


MADE = '\n[cases.made]\nrows = 2\ns = 2.0\ngenerator = "gen.py:make"\n'

# --- expansion ----------------------------------------------------------------------------


def test_scale_expands_to_seven_cases(tmp_path):
    cases = expand(scale(tmp_path))
    assert [c.name for c in cases] == [
        "default",
        "one_row",
        "sweep[rows=1]",
        "sweep[rows=7]",
        "sweep[rows=33]",
        "sweep[rows=128]",
        "mine",
    ]
    sweep = cases[3]
    assert dict(sweep.values) == {"rows": 7, "s": 1.5}
    assert sweep.formats == (F.FP32, F.E4) and cases[0].formats == (F.FP32, F.INT8, F.E4)
    assert sweep.decl.name == "sweep"


def test_a_two_axis_sweep_is_the_cartesian_product_in_order(tmp_path):
    m = scale(tmp_path, extra_cases="\n[cases.grid]\nrows = [2, 1]\ns = [0.5, 1.0, 2.0]\n")
    names = [c.name for c in expand(m) if c.name.startswith("grid")]
    assert names == [
        "grid[rows=2,s=0.5]",
        "grid[rows=2,s=1.0]",
        "grid[rows=2,s=2.0]",
        "grid[rows=1,s=0.5]",
        "grid[rows=1,s=1.0]",
        "grid[rows=1,s=2.0]",
    ]


def test_a_value_the_case_does_not_give_takes_its_default(tmp_path):
    m = scale(tmp_path, extra_cases="\n[cases.half]\ns = 0.5\n")
    assert dict(next(c for c in expand(m) if c.name == "half").values) == {"rows": 4, "s": 0.5}


# --- the random streams -------------------------------------------------------------------

_RNG = (
    "import json; from ipu_harness.cases import rng; "
    "print(json.dumps(rng('k', 'c', 'x').integers(0, 2**31, 4).tolist()))"
)


def _rng_in_subprocess(hashseed: str) -> list[int]:
    env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run(
        [sys.executable, "-c", _RNG], env=env, capture_output=True, text=True, check=True
    )
    return list(json.loads(done.stdout))


def test_rng_is_stable_across_processes_and_independent_per_tensor():
    first = _rng_in_subprocess("1")
    assert first == _rng_in_subprocess("2") == rng("k", "c", "x").integers(0, 2**31, 4).tolist()
    assert rng("k", "c", "x").random(8).tolist() != rng("k", "c", "w").random(8).tolist()
    assert rng("k", "c", "x").random(8).tolist() != rng("k", "d", "x").random(8).tolist()


# --- the default generator ----------------------------------------------------------------


def test_generated_inputs_are_in_the_format_and_reproducible(tmp_path):
    m = scale(tmp_path)
    case, fp32 = planned(m, "default")
    x = inputs_for(case, fp32).tensors["x"]
    assert x.dtype == np.float32 and x.shape == (4, 128) and -1 <= x.min() and x.max() <= 1
    assert np.array_equal(x, inputs_for(case, fp32).tensors["x"])
    _, int8 = planned(m, "default", F.INT8)
    xi = inputs_for(case, int8).tensors["x"]
    assert xi.dtype == np.int8 and xi.min() < -100 and xi.max() > 100  # the whole INT8 range
    _, e4 = planned(m, "default", F.E4)
    xe = inputs_for(case, e4).tensors["x"]
    # E4 holds 3 mantissa bits: each value in [0.5, 1) is a multiple of 1/16.
    upper = xe[(xe >= 0.5) & (xe < 1)]
    assert upper.size and np.array_equal(upper * 16, np.round(upper * 16))
    assert inputs_for(case, e4).rounded == {}


def test_range_keeps_generated_values_inside_it_per_format(tmp_path):
    m = scale(tmp_path, x_extra="range = { int8 = [-3, 3], fp32 = [0, 0.25] }\n")
    case, int8 = planned(m, "default", F.INT8)
    xi = inputs_for(case, int8).tensors["x"]
    assert set(np.unique(xi).tolist()) == {-3, -2, -1, 0, 1, 2, 3}
    _, fp32 = planned(m, "default")
    x = inputs_for(case, fp32).tensors["x"]
    assert 0 <= x.min() and x.max() <= 0.25
    _, e4 = planned(m, "default", F.E4)  # not listed: the default range
    assert inputs_for(case, e4).tensors["x"].min() < 0


# --- the sources and their priority -------------------------------------------------------


def test_input_priority_override_then_case_file_then_generator(tmp_path):
    both = MADE.replace(
        'generator = "gen.py:make"', 'inputs = { x = "data/x.npy" }\ngenerator = "gen.py:make"'
    )
    m = scale(tmp_path, extra_cases=both.replace("made]", "both]") + MADE)
    generator(m, "return {'x': np.full(plan.shapes['x'], 0.5)}")
    file_x = np.load(m.directory / "data" / "x.npy")
    case, plan = planned(m, "both")
    assert np.array_equal(inputs_for(case, plan).tensors["x"], file_x)  # file beats generator
    override = tmp_path / "override.npy"
    np.save(override, np.ones((2, 128)))
    assert np.array_equal(
        inputs_for(case, plan, overrides={"x": override}).tensors["x"], np.ones((2, 128))
    )
    case, plan = planned(m, "made")  # no file: the generator's
    assert np.array_equal(inputs_for(case, plan).tensors["x"], np.full((2, 128), 0.5))


def test_a_generators_values_are_rounded_to_the_format_and_counted(tmp_path):
    m = scale(tmp_path, extra_cases=MADE)
    generator(m, "return {'x': np.full(plan.shapes['x'], 0.3)}")
    case, e4 = planned(m, "made", F.E4)
    made = inputs_for(case, e4)
    assert np.array_equal(made.tensors["x"], np.full((2, 128), E4_NEAR_0_3, dtype=np.float32))
    assert made.rounded == {"x": 256}


def test_a_generator_may_be_the_familys_shared_file(tmp_path):
    """``../gen.py:make``: one generator for every kernel of a family (E07-02)."""
    m = scale(tmp_path / "family", extra_cases=MADE.replace('"gen.py:make"', '"../gen.py:make"'))
    (m.directory.parent / "gen.py").write_text(
        "import numpy as np\n\n\ndef make(plan, rng):\n    return {'x': np.full(plan.shapes['x'], 0.5)}\n"
    )
    case, fp32 = planned(m, "made")
    assert np.array_equal(inputs_for(case, fp32).tensors["x"], np.full((2, 128), 0.5))


def test_a_user_file_is_rounded_to_the_format_and_counted(tmp_path):
    # E06-10's rule: user data is rounded (saturating) when packed, and the
    # report says how many values changed; it is not rejected.
    m = scale(tmp_path)
    x = np.zeros((2, 128))
    x[0, :3] = (1.5, 200.0, -300.0)  # INT8: 1.5 -> 2 (to even), 200 -> 127, -300 -> -128
    np.save(m.directory / "data" / "x.npy", x)
    case, int8 = planned(m, "mine", F.INT8)
    got = inputs_for(case, int8)
    assert got.tensors["x"][0, :4].tolist() == [2, 127, -128, 0]
    assert got.rounded == {"x": 3}
    _, fp32 = planned(m, "mine")
    assert inputs_for(case, fp32).rounded == {}  # exact in FP32


def test_an_override_names_an_input_and_its_errors_name_the_file(tmp_path):
    case, plan = planned(scale(tmp_path), "default")
    with pytest.raises(ConfigurationError, match="'z' is no input"):
        inputs_for(case, plan, overrides={"z": tmp_path / "z.npy"})
    bad = tmp_path / "bad.npy"
    np.save(bad, np.full((4, 128), np.nan))
    with pytest.raises(
        ConfigurationError, match=r"--input x=.*bad\.npy: x holds a value that is not finite"
    ):
        inputs_for(case, plan, overrides={"x": bad})


@pytest.mark.parametrize(
    "value, message",
    [
        (np.zeros((3, 128)), r"x has shape \(3, 128\), the case's is \(2, 128\)"),
        (np.array([["a"] * 128] * 2), "dtype <U1, not an int or float one"),
        (np.zeros((2, 128), dtype=np.complex64), "dtype complex64, not an int or float one"),
        (np.zeros((2, 128), dtype=bool), "dtype bool, not an int or float one"),
        (np.full((2, 128), np.inf), "not finite"),
    ],
)
def test_a_case_file_of_the_wrong_shape_dtype_or_values_is_rejected(tmp_path, value, message):
    m = scale(tmp_path)
    np.save(m.directory / "data" / "x.npy", value)
    case, plan = planned(m, "mine")
    with pytest.raises(PlanError, match=message) as error:
        inputs_for(case, plan)
    assert error.value.key == "cases.mine.inputs.x"


def test_an_object_npy_is_rejected(tmp_path):
    m = scale(tmp_path)
    objects = np.array([object()] * 256).reshape(2, 128)
    np.save(m.directory / "data" / "x.npy", objects, allow_pickle=True)
    case, plan = planned(m, "mine")
    with pytest.raises(PlanError, match="cannot read") as error:
        inputs_for(case, plan)
    assert error.value.key == "cases.mine.inputs.x"


@pytest.mark.parametrize(
    "body, message",
    [
        ("return {'x': b'\\x00' * 256}", "x is bytes, not an array"),
        ("return {'w': np.zeros((2, 128))}", "made 'w', which is no input"),
        ("return [np.zeros((2, 128))]", "returned list, not a dict"),
        ("return {'x': np.full((2, 128), np.nan)}", "not finite"),
        ("raise RuntimeError('boom')", "raised RuntimeError: boom"),
    ],
)
def test_a_generator_returns_finite_arrays_of_declared_inputs(tmp_path, body, message):
    m = scale(tmp_path, extra_cases=MADE)
    generator(m, body)
    case, plan = planned(m, "made")
    with pytest.raises(PlanError, match=message) as error:
        inputs_for(case, plan)
    assert error.value.key == "cases.made.generator"


def test_a_generator_file_that_does_not_load_or_lacks_the_function_is_located(tmp_path):
    m = scale(tmp_path, extra_cases=MADE)
    (m.directory / "gen.py").write_text("def make(:\n")
    case, plan = planned(m, "made")
    with pytest.raises(PlanError, match="cannot load gen.py: SyntaxError") as error:
        inputs_for(case, plan)
    assert error.value.key == "cases.made.generator"
    (m.directory / "gen.py").write_text("def other(plan, rng):\n    return {}\n")
    with pytest.raises(PlanError, match="gen.py has no function make"):
        inputs_for(case, plan)
