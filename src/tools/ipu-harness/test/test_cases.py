"""Cases: sweeps, seeded generators, user data and its priority (E06-08)."""

from __future__ import annotations

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

HARNESS = Path(__file__).resolve().parents[1] / "src" / "ipu_harness"

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


def scale(tmp_path: Path, *, x_extra: str = "", extra_cases: str = "") -> KernelManifest:
    directory = tmp_path / "scale"
    (directory / "data").mkdir(parents=True)
    (directory / "scale.asm").write_text("HALT ;;\n")
    np.save(
        directory / "data" / "x.npy", np.arange(-4, 4, dtype=np.float32).repeat(32).reshape(2, 128)
    )
    np.save(directory / "data" / "y.npy", np.zeros((2, 128), dtype=np.float32))
    (directory / "kernel.toml").write_text(SCALE.format(x_extra=x_extra, extra_cases=extra_cases))
    return load_manifest(directory)


def planned(manifest: KernelManifest, name: str, fmt: F = F.FP32):
    case = next(c for c in expand(manifest) if c.name == name)
    return case, plan_case(manifest, case.values, fmt, case=case.decl)


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
    assert names == [f"grid[rows={r},s={s}]" for r in (2, 1) for s in (0.5, 1.0, 2.0)]


def test_a_value_the_case_does_not_give_takes_its_default(tmp_path):
    m = scale(tmp_path, extra_cases="\n[cases.half]\ns = 0.5\n")
    assert dict(next(c for c in expand(m) if c.name == "half").values) == {"rows": 4, "s": 0.5}


_RNG = "from ipu_harness.cases import rng; print(rng('k', 'c', 'x').integers(0, 2**31, 8).tolist())"


def _rng_in_subprocess(hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run(
        [sys.executable, "-c", _RNG], env=env, capture_output=True, text=True, check=True
    )
    return done.stdout


def test_rng_is_stable_across_processes_and_independent_per_tensor():
    assert _rng_in_subprocess("1") == _rng_in_subprocess("2")
    assert rng("k", "c", "x").integers(0, 2**31, 8).tolist() == eval(_rng_in_subprocess("3"))
    assert rng("k", "c", "x").random(8).tolist() != rng("k", "c", "w").random(8).tolist()


def test_generated_inputs_are_in_the_format_and_reproducible(tmp_path):
    m = scale(tmp_path)
    case, fp32 = planned(m, "default")
    x = inputs_for(case, fp32)["x"]
    assert x.dtype == np.float32 and x.shape == (4, 128) and -1 <= x.min() and x.max() < 1
    assert np.array_equal(x, inputs_for(case, fp32)["x"])
    _, int8 = planned(m, "default", F.INT8)
    xi = inputs_for(case, int8)["x"]
    assert xi.dtype == np.int8 and xi.min() < -100 and xi.max() > 100  # the whole INT8 range
    _, e4 = planned(m, "default", F.E4)
    xe = inputs_for(case, e4)["x"]
    from ipu_harness.params import representable

    assert all(representable(float(v), F.E4)[0] for v in xe.ravel()[:64])


def test_range_keeps_generated_values_inside_it_per_format(tmp_path):
    m = scale(tmp_path, x_extra="range = { int8 = [-3, 3], fp32 = [0, 0.25] }\n")
    case, int8 = planned(m, "default", F.INT8)
    xi = inputs_for(case, int8)["x"]
    assert xi.min() >= -3 and xi.max() <= 3 and set(np.unique(xi)) == set(range(-3, 4))
    _, fp32 = planned(m, "default")
    x = inputs_for(case, fp32)["x"]
    assert 0 <= x.min() and x.max() < 0.25
    _, e4 = planned(m, "default", F.E4)  # not listed: the default range
    assert inputs_for(case, e4)["x"].min() < 0


def test_input_priority_override_then_case_file_then_generator(tmp_path):
    m = scale(
        tmp_path,
        extra_cases='\n[cases.made]\nrows = 2\ns = 2.0\ninputs = { x = "data/x.npy" }\n'
        'generator = "gen.py:make"\n',
    )
    (m.directory / "gen.py").write_text(
        "import numpy as np\n"
        "def make(plan, rng):\n"
        "    return {'x': np.full(plan.shapes['x'], 0.5)}\n"
    )
    file_x = np.load(m.directory / "data" / "x.npy")
    case, plan = planned(m, "made")
    assert np.array_equal(inputs_for(case, plan)["x"], file_x)  # the case file beats the generator
    override = tmp_path / "override.npy"
    np.save(override, np.ones((2, 128)))
    assert np.array_equal(inputs_for(case, plan, overrides={"x": override})["x"], np.ones((2, 128)))
    # Without a file the generator's tensor is used, rounded to the format.
    (m.directory / "kernel.toml").write_text(
        (m.directory / "kernel.toml")
        .read_text()
        .replace('inputs = { x = "data/x.npy" }\ngen', "gen")
    )
    m2 = load_manifest(m.directory)
    case2, plan2 = planned(m2, "made")
    assert np.array_equal(inputs_for(case2, plan2)["x"], np.full((2, 128), 0.5, dtype=np.float32))


def test_an_override_names_an_input(tmp_path):
    case, plan = planned(scale(tmp_path), "default")
    with pytest.raises(ConfigurationError, match="'z' is no input"):
        inputs_for(case, plan, overrides={"z": tmp_path / "z.npy"})


@pytest.mark.parametrize(
    "value, message",
    [
        (np.zeros((3, 128)), r"has shape \(3, 128\), the case's is \(2, 128\)"),
        (np.array([["a"] * 128] * 2), "not a numeric one"),
        (np.full((2, 128), 0.1), "fp32 cannot hold exactly"),
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
    np.save(
        m.directory / "data" / "x.npy",
        np.array([object()] * 256).reshape(2, 128),
        allow_pickle=True,
    )
    case, plan = planned(m, "mine")
    with pytest.raises(PlanError, match="cannot read"):
        inputs_for(case, plan)


@pytest.mark.parametrize(
    "body, message",
    [
        ("return {'x': b'\\x00' * 256}", "x is bytes, not an array"),
        ("return {'w': np.zeros((2, 128))}", "made 'w', which is no input"),
        ("return [np.zeros((2, 128))]", "returned list, not a dict"),
    ],
)
def test_a_generator_returns_declared_inputs_as_arrays(tmp_path, body, message):
    m = scale(
        tmp_path, extra_cases='\n[cases.made]\nrows = 2\ns = 2.0\ngenerator = "gen.py:make"\n'
    )
    (m.directory / "gen.py").write_text(f"import numpy as np\ndef make(plan, rng):\n    {body}\n")
    case, plan = planned(m, "made")
    with pytest.raises(PlanError, match=message) as error:
        inputs_for(case, plan)
    assert error.value.key == "cases.made.generator"


def test_no_per_call_reseeding_or_hash_seeding_in_the_harness():
    for path in sorted(HARNESS.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "hash(" not in text, path
        assert "RandomState" not in text, path
        assert "default_rng(" not in text or path.name == "cases.py", path
