"""Golden comparison: expected values held by the variant, named tiers (E06-10, ADR 0007)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_harness.cases import expand
from ipu_harness.compare import compare, expected_for, nx8_steps, tier_for
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.params import PlanError, plan_case
from ipu_harness.tolerance import TIERS
from ipu_harness.variants import Variant, variant

FP32, E4 = variant("fp32"), variant("e4")

TOML = """\
[kernel]
op = "scale"
formats = ["fp32", "e3", "e4"]

[params]
rows = {{ type = "int", default = 1 }}
s    = {{ type = "float", default = 2.0 }}

[inputs.x]
shape = ["rows", 4]

[outputs.y]
shape = ["rows", 4]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[golden]
tolerance = {{ fp32 = "FP32_ORDER", e4 = "EXACT" }}

[cases.default]

[cases.mine]
expected = {{ y = "y.npy" }}
"""


def kernel(tmp_path: Path, reference: bool = True) -> KernelManifest:
    directory = tmp_path / "elementwise" / "scale"
    directory.mkdir(parents=True)
    (directory / "scale.asm").write_text("HALT ;;\n")
    if reference:
        (directory / "reference.py").write_text("def reference(x, s):\n    return {'y': x * s}\n")
    np.save(directory / "y.npy", np.array([[1.0, 2.0, 3.0, 4.0]]))
    (directory / "kernel.toml").write_text(TOML.format())
    return load_manifest(directory)


def planned(m: KernelManifest, name: str, fmt: Variant):
    case = next(c for c in expand(m) if c.name == name)
    return case, plan_case(m, case.values, fmt, case=case.decl)


def test_nx8_expected_values_are_quantized_row_by_row(tmp_path):
    # Two rows: each row's scale follows its largest value.
    m = kernel(tmp_path)
    case = next(c for c in expand(m) if c.name == "default")
    plan = plan_case(m, {"rows": 2}, E4, case=case.decl)
    reference = {"y": np.array([[1.0625, 1.1875, 1.125, 1.875], [1.0, 2**-7, 3.0, 1000.0]])}
    want = expected_for(case, plan, reference)["y"]
    # Row 0 (scale 2^0): E4 holds 1.0, 1.125, 1.25 near 1 (3 mantissa bits), and
    # ties round half up (ARCH-stage-aaq-str.md 4.3.2). Row 1 (scale 2^9, from
    # 1000): 2^-7 is a subnormal step there, and 1000 saturates to 960 (1.875 x 2^9).
    assert want.tolist() == [[1.125, 1.25, 1.125, 1.875], [1.0, 2**-7, 3.0, 960.0]]
    # A kernel that quantizes the same way passes at EXACT.
    result = compare("y", want.copy(), want, tier_for(plan, plan.manifest.tensors["y"]), E4)
    assert result.passed and result.tier.name == "EXACT" and result.max_abs == 0


def test_one_nx8_step_is_the_golden_rows_grid_spacing(tmp_path):
    m = kernel(tmp_path)
    case, plan = planned(m, "default", E4)
    want = expected_for(case, plan, {"y": np.array([[1.0, 2.0, 0.5, 1.5]])})["y"]
    steps = nx8_steps(m.tensors["y"], want, plan)
    # Scale 2^1 (from 2.0): each element's larger gap to a grid neighbour. 1.0
    # sits between 0.9375 and 1.125, 2.0 between 1.875 and 2.25, 0.5 between
    # 0.46875 and 0.5625, 1.5 between 1.375 and 1.625.
    assert steps.tolist() == [[0.125, 0.25, 0.0625, 0.125]]


def test_a_planted_error_is_named_by_output_index_values_and_tier(tmp_path):
    want = np.zeros((2, 4))
    got = want.copy()
    got[1, 2] = 0.5
    result = compare("y", got, want, TIERS["FP32_ORDER"], FP32)
    assert not result.passed
    assert (result.output, result.worst_index, result.got, result.want) == ("y", (1, 2), 0.5, 0.0)
    assert result.tier.name == "FP32_ORDER" and result.max_abs == 0.5
    assert "y: FAIL at FP32_ORDER" in result.describe() and "[1, 2]" in result.describe()


def test_tiers_hold_by_tolerance_and_by_nx8_steps(tmp_path):
    want = np.array([1.0, 100.0])
    assert compare("y", want * (1 + 9e-6), want, TIERS["FP32_ORDER"], FP32).passed
    assert not compare("y", want * (1 + 2e-5), want, TIERS["FP32_ORDER"], FP32).passed
    # One step of 0.125 next to 1.0: 1.125 is one step away, 1.25 two.
    step = np.array([0.125])
    one = compare("y", np.array([1.125]), np.array([1.0]), TIERS["NX8_STEP"], E4, step)
    two = compare("y", np.array([1.25]), np.array([1.0]), TIERS["NX8_STEP"], E4, step)
    assert (one.passed, one.max_ulps, two.passed, two.max_ulps) == (True, 1, False, 2)
    with pytest.raises(ValueError, match="counts NX8 steps"):
        compare("y", np.array([1.0]), np.array([1.0]), TIERS["NX8_STEP"], E4)


def test_an_expected_file_wins_over_the_reference(tmp_path):
    m = kernel(tmp_path)
    assert m.golden.reference == "reference.py:reference"
    case, plan = planned(m, "mine", FP32)
    want = expected_for(case, plan, {"y": np.full((1, 4), 9.0)})["y"]
    assert want.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_an_expected_function_gives_the_values_for_the_plan(tmp_path):
    """A golden kept in another encoding (fully_connected's .bin, E07-21)."""
    m = kernel(tmp_path)
    (m.directory / "gold.py").write_text(
        "import numpy as np\n\n"
        "def y(plan):\n"
        "    return np.full(plan.shapes['y'], 3.0 if plan.format.value == 'fp32' else 4.0)\n"
    )
    toml = (m.directory / "kernel.toml").read_text()
    (m.directory / "kernel.toml").write_text(
        toml.replace('expected = { y = "y.npy" }', 'expected = { y = "gold.py:y" }')
    )
    m = load_manifest(m.directory)
    assert m.cases["mine"].expected == {"y": "gold.py:y"}
    for fmt, value in ((FP32, 3.0), (E4, 4.0)):
        case, plan = planned(m, "mine", fmt)
        assert expected_for(case, plan, {"y": np.zeros((1, 4))})["y"].tolist() == [[value] * 4]


def test_an_expected_function_that_fails_is_located(tmp_path):
    m = kernel(tmp_path)
    (m.directory / "gold.py").write_text("def y(plan):\n    raise FileNotFoundError('y.bin')\n")
    toml = (m.directory / "kernel.toml").read_text()
    (m.directory / "kernel.toml").write_text(
        toml.replace('expected = { y = "y.npy" }', 'expected = { y = "gold.py:y" }')
    )
    case, plan = planned(load_manifest(m.directory), "mine", FP32)
    with pytest.raises(PlanError, match=r"gold.py:y raised FileNotFoundError: y.bin") as error:
        expected_for(case, plan, None)
    assert error.value.key == "cases.mine.expected.y"


def test_no_reference_and_no_file_is_no_golden(tmp_path):
    case, plan = planned(kernel(tmp_path, reference=False), "default", FP32)
    assert expected_for(case, plan, None) == {}
    _, e3 = planned(kernel(tmp_path / "b"), "default", variant("e3"))
    assert tier_for(e3, e3.manifest.tensors["y"]).name == "EXACT"  # no tier named for e3


def test_an_expected_file_or_reference_of_the_wrong_shape_is_located(tmp_path):
    m = kernel(tmp_path)
    np.save(m.directory / "y.npy", np.zeros((2, 4)))
    case, plan = planned(m, "mine", FP32)
    with pytest.raises(PlanError, match=r"holds float64 \(2, 4\), not \(1, 4\)") as error:
        expected_for(case, plan, None)
    assert error.value.key == "cases.mine.expected.y"
    case, plan = planned(m, "default", FP32)
    with pytest.raises(PlanError, match=r"made y \(3,\), not \(1, 4\)"):
        expected_for(case, plan, {"y": np.zeros(3)})


def test_a_non_finite_value_never_passes_at_nx8_step():
    tier = TIERS["NX8_STEP"]
    step = np.array([0.125, 0.125])
    for bad in (np.nan, np.inf):
        result = compare("y", np.array([bad, 1.0]), np.array([1.125, 1.0]), tier, E4, step)
        assert not result.passed and result.max_ulps == -1 and result.worst_index == (0,)
    # Between two grid values but within one step of the golden: it passes.
    assert compare("y", np.array([1.06, 1.0]), np.array([1.125, 1.0]), tier, E4, step).passed


def test_nan_fails_a_tolerance_tier_and_an_empty_output_passes():
    assert not compare("y", np.array([np.nan]), np.array([1.0]), TIERS["FP32_LONG"], FP32).passed
    assert compare("y", np.zeros((0, 4)), np.zeros((0, 4)), TIERS["EXACT"], FP32).passed


def test_the_reference_must_make_exactly_the_declared_outputs(tmp_path):
    case, plan = planned(kernel(tmp_path), "default", FP32)
    for made in ({}, {"y": np.zeros((1, 4)), "z": np.zeros(1)}, {"Y": np.zeros((1, 4))}):
        with pytest.raises(PlanError, match="not the outputs \\['y'\\]") as error:
            expected_for(case, plan, made)
        assert error.value.key == "golden"
    with pytest.raises(PlanError, match="non-finite"):
        expected_for(case, plan, {"y": np.array([[1.0, np.nan, 0.0, 0.0]])})
