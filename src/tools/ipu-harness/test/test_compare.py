"""Golden comparison: expected values rounded to the format, named tiers (E06-10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_harness.cases import expand
from ipu_harness.compare import compare, expected_for, tier_for
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.params import PlanError, plan_case
from ipu_harness.tolerance import TIERS
from ipu_isa.formats import ElementFormat as F

TOML = """\
[kernel]
op = "scale"
formats = ["fp32", "int8", "e4"]

[params]
rows = {{ type = "int", default = 1 }}
s    = {{ type = "float", default = 2.0 }}

[inputs.x]
shape = ["rows", 4]

[outputs.y]
shape = ["rows", 4]
{store}
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


def kernel(tmp_path: Path, store: str = "", reference: bool = True) -> KernelManifest:
    directory = tmp_path / "elementwise" / "scale"
    directory.mkdir(parents=True)
    (directory / "scale.asm").write_text("HALT ;;\n")
    if reference:
        (directory / "reference.py").write_text("def reference(x, s):\n    return {'y': x * s}\n")
    np.save(directory / "y.npy", np.array([[1.0, 2.0, 3.0, 4.0]]))
    (directory / "kernel.toml").write_text(TOML.format(store=store))
    return load_manifest(directory)


def planned(m: KernelManifest, name: str, fmt: F):
    case = next(c for c in expand(m) if c.name == name)
    return case, plan_case(m, case.values, fmt, case=case.decl)


def test_e4_expected_values_round_to_the_nearest_even_code(tmp_path):
    case, plan = planned(kernel(tmp_path), "default", F.E4)
    # E4 holds 1.0, 1.125, 1.25 near 1 (3 mantissa bits): the midpoints round to even.
    reference = {"y": np.array([[1.0625, 1.1875, 1.125, 1e6]])}
    want = expected_for(case, plan, reference)["y"]
    assert want.tolist() == [[1.0, 1.25, 1.125, 240.0]]  # 1e6 saturates to E4's largest
    # A kernel that rounds the same way passes at EXACT.
    result = compare("y", want.copy(), want, tier_for(plan), F.E4)
    assert result.passed and result.tier.name == "EXACT" and result.max_abs == 0


def test_int8_expected_values_saturate(tmp_path):
    case, plan = planned(kernel(tmp_path), "default", F.INT8)
    want = expected_for(case, plan, {"y": np.array([[100 * 2.0, -100 * 2.0, 2.5, 3.5]])})["y"]
    assert want.tolist() == [[127, -128, 2, 4]]


def test_an_accumulator_output_is_rounded_to_its_accumulator_format(tmp_path):
    case, plan = planned(kernel(tmp_path, store='store = "accumulator"\n'), "default", F.INT8)
    want = expected_for(case, plan, {"y": np.array([[200.0, -300.4, 2.5, 1e10]])})["y"]
    assert want.tolist() == [[200, -300, 2, 2**31 - 1]]  # INT32, not INT8


def test_a_planted_error_is_named_by_output_index_values_and_tier(tmp_path):
    want = np.zeros((2, 4))
    got = want.copy()
    got[1, 2] = 0.5
    result = compare("y", got, want, TIERS["FP32_ORDER"], F.FP32)
    assert not result.passed
    assert (result.output, result.worst_index, result.got, result.want) == ("y", (1, 2), 0.5, 0.0)
    assert result.tier.name == "FP32_ORDER" and result.max_abs == 0.5
    assert "y: FAIL at FP32_ORDER" in result.describe() and "[1, 2]" in result.describe()


def test_tiers_hold_by_tolerance_and_by_code_distance(tmp_path):
    want = np.array([1.0, 100.0])
    assert compare("y", want * (1 + 9e-6), want, TIERS["FP32_ORDER"], F.FP32).passed
    assert not compare("y", want * (1 + 2e-5), want, TIERS["FP32_ORDER"], F.FP32).passed
    assert compare(
        "y", np.array([3.0, -127.0]), np.array([2.0, -128.0]), TIERS["INT8_LSB"], F.INT8
    ).passed
    # E4 codes next to 1.0: 1.125 is one code away, 1.25 two.
    one = compare("y", np.array([1.125]), np.array([1.0]), TIERS["FP8_ULP"], F.E4)
    two = compare("y", np.array([1.25]), np.array([1.0]), TIERS["FP8_ULP"], F.E4)
    assert (one.passed, one.max_ulps, two.passed, two.max_ulps) == (True, 1, False, 2)


def test_an_expected_file_wins_over_the_reference(tmp_path):
    m = kernel(tmp_path)
    assert m.golden.reference == "reference.py:reference"
    case, plan = planned(m, "mine", F.FP32)
    want = expected_for(case, plan, {"y": np.full((1, 4), 9.0)})["y"]
    assert want.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_no_reference_and_no_file_is_no_golden(tmp_path):
    case, plan = planned(kernel(tmp_path, reference=False), "default", F.FP32)
    assert expected_for(case, plan, None) == {}
    assert tier_for(planned(kernel(tmp_path / "b"), "default", F.INT8)[1]).name == "EXACT"


def test_an_expected_file_or_reference_of_the_wrong_shape_is_located(tmp_path):
    m = kernel(tmp_path)
    np.save(m.directory / "y.npy", np.zeros((2, 4)))
    case, plan = planned(m, "mine", F.FP32)
    with pytest.raises(PlanError, match=r"holds float64 \(2, 4\), not \(1, 4\)") as error:
        expected_for(case, plan, None)
    assert error.value.key == "cases.mine.expected.y"
    case, plan = planned(m, "default", F.FP32)
    with pytest.raises(PlanError, match=r"made y \(3,\), not \(1, 4\)"):
        expected_for(case, plan, {"y": np.zeros(3)})
