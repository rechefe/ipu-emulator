"""plan_case: one test per check, shape expressions, element scalars (E06-03)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_harness import params as params_module
from ipu_harness.manifest import load_manifest
from ipu_harness.params import (
    CasePlan,
    ExpressionError,
    PlanError,
    check_params,
    evaluate,
    plan_case,
    representable,
)
from ipu_isa.formats import ElementFormat as F

TOML = """\
[kernel]
op = "scale"
formats = ["fp32", "int8", "e4"]

[params]
rows = { type = "int", min = 1, max = 256 }
k    = { type = "int", min = 1, multiple_of = 4, default = 128 }
mode = { type = "str", choices = ["a", "b"], default = "a" }
s    = { type = "float" }

[derived]
blocks = "ceil(k / 128)"

[inputs.x]
shape = ["rows", "k"]

[outputs.y]
shape = ["rows * 4", "blocks"]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[golden]
op = "scale"

[cases.default]
rows = 4
s = 2.0

[cases.fp32_only]
rows = 1
s = 1.5
formats = ["fp32"]
"""


@pytest.fixture
def manifest(tmp_path: Path):
    directory = tmp_path / "scale"
    directory.mkdir()
    (directory / "scale.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(TOML)
    return load_manifest(directory)


def test_a_plan_has_complete_params_and_int_shapes(manifest):
    plan = plan_case(manifest, {"rows": 3, "s": 2}, F.FP32)
    assert isinstance(plan, CasePlan) and plan.format is F.FP32
    assert dict(plan.params) == {"rows": 3, "k": 128, "mode": "a", "s": 2.0}
    assert isinstance(plan.params["s"], float)
    assert dict(plan.shapes) == {"x": (3, 128), "y": (12, 1)}
    assert dict(plan.derived) == {"blocks": 1}
    with pytest.raises(TypeError):
        plan.params["rows"] = 4  # type: ignore[index]


@pytest.mark.parametrize(
    ("values", "fmt", "case", "key", "words"),
    [
        # 1. format
        ({"rows": 1, "s": 1.0}, F.E5, None, "formats", "format e5 is not one of the kernel's"),
        ({"rows": 1, "s": 1.0}, F.INT8, "fp32_only", "cases.fp32_only.formats", "format int8"),
        # 2. parameters
        ({"rows": 1, "s": 1.0, "cols": 2}, F.FP32, None, "params.cols", "cols = 2 is no parameter"),
        ({"rows": 1}, F.FP32, None, "params.s", "s has no default"),
        ({"rows": "1", "s": 1.0}, F.FP32, None, "params.rows", "rows = '1' is not int"),
        ({"rows": True, "s": 1.0}, F.FP32, None, "params.rows", "rows = True is not int"),
        ({"rows": 0, "s": 1.0}, F.FP32, None, "params.rows", "rows = 0 is below its min 1"),
        ({"rows": 257, "s": 1.0}, F.FP32, None, "params.rows", "rows = 257 is above its max 256"),
        ({"rows": 1, "k": 6, "s": 1.0}, F.FP32, None, "params.k", "k = 6 is not a multiple of 4"),
        (
            {"rows": 1, "mode": "c", "s": 1.0},
            F.FP32,
            None,
            "params.mode",
            "mode = 'c' is not one of",
        ),
        # 4. element scalars
        (
            {"rows": 1, "s": 1.5},
            F.INT8,
            None,
            "params.s",
            "s = 1.5 does not encode exactly in int8",
        ),
        ({"rows": 1, "s": 0.1}, F.E4, None, "params.s", "s = 0.1 does not encode exactly in e4"),
        (
            {"rows": 1, "s": 0.1},
            F.FP32,
            None,
            "params.s",
            "s = 0.1 does not encode exactly in fp32",
        ),
    ],
)
def test_each_check_names_the_key_and_the_value(manifest, values, fmt, case, key, words):
    with pytest.raises(PlanError) as error:
        plan_case(manifest, values, fmt, case=case)
    assert error.value.key == key
    assert words in str(error.value)
    assert str(error.value).startswith(f"{manifest.directory / 'kernel.toml'}: {key}: ")


def test_a_shape_that_is_not_positive_is_an_error(manifest):
    toml = TOML.replace('shape = ["rows * 4", "blocks"]', 'shape = ["rows - 4", "blocks"]')
    (manifest.directory / "kernel.toml").write_text(toml)
    with pytest.raises(PlanError, match=r"outputs.y.shape: 'rows - 4' = -1, not a positive size"):
        plan_case(load_manifest(manifest.directory), {"rows": 3, "s": 2.0}, F.FP32)


def test_element_scalars_exact_in_the_format_pass():
    assert representable(2.0, F.INT8) == (True, ())
    assert representable(2.0, F.E4) == (True, ())
    assert representable(1.5, F.FP32) == (True, ())
    assert representable(1.5, F.INT8) == (False, (1.0, 2.0))
    exact, near = representable(0.1, F.E4)
    assert not exact and len(near) == 2 and near[0] < 0.1 < near[1]


def test_s_1_5_is_rejected_in_int8_and_2_0_accepted_in_int8_and_e4(manifest):
    with pytest.raises(PlanError, match="nearest: 1.0 and 2.0"):
        plan_case(manifest, {"rows": 1, "s": 1.5}, F.INT8)
    for fmt in (F.INT8, F.E4):
        assert plan_case(manifest, {"rows": 1, "s": 2.0}, fmt).params["s"] == 2.0


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        ("ceil(k / 128)", 2),
        ("rows * 4", 12),
        ("k // 7 % 5", 3),
        ("max(rows, 2) - min(1, k)", 2),
        ("-rows + 5", 2),
    ],
)
def test_shape_expressions_evaluate(expression, value):
    assert evaluate(expression, {"k": 129, "rows": 3}) == value


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "rows.real",
        "abs(rows)",
        "ceil(x=rows)",
        "rows ** 2",
        "k / 3",  # not a whole number
        "rows / 0",
        "1.5 * rows",
        "'4'",
        "unknown",
        "rows if rows else 1",
        "[rows]",
        "rows +",
    ],
)
def test_anything_else_is_rejected(expression):
    with pytest.raises(ExpressionError):
        evaluate(expression, {"k": 128, "rows": 3})


def test_check_params_is_the_same_checks_without_a_format(manifest):
    check_params(manifest, {"rows": 2, "s": 0.1})  # no format: s is not checked against one
    with pytest.raises(PlanError, match="rows = 0 is below its min 1"):
        check_params(manifest, {"rows": 0, "s": 1.0})


def test_plan_case_is_the_only_validation(monkeypatch, manifest):
    # Every check is a function plan_case calls once per case run, and nothing else in the
    # package calls them: a later stage re-validating would show up here.
    calls = []
    real = params_module._check_params
    monkeypatch.setattr(params_module, "_check_params", lambda *a: calls.append(a) or real(*a))
    plan_case(manifest, {"rows": 2, "s": 2.0}, F.FP32)
    assert len(calls) == 1
    package = Path(params_module.__file__).parent
    others = [p for p in package.glob("*.py") if p.name != "params.py"]
    assert not [
        p.name
        for p in others
        if "_check_params" in p.read_text() or "_check_value" in p.read_text()
    ]
