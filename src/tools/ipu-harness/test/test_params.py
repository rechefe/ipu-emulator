"""plan_case: one test per check, shape expressions, element scalars (E06-03)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ipu_harness.expressions import ExpressionError, evaluate
from ipu_harness.manifest import load_manifest
from ipu_harness.params import CasePlan, PlanError, check_params, plan_case, representable
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
        plan_case(manifest, values, fmt, case=manifest.cases[case] if case else None)
    assert error.value.key == key
    assert words in str(error.value)
    assert str(error.value).startswith(f"{manifest.directory / 'kernel.toml'}: {key}: ")


def test_a_shape_that_is_not_positive_is_an_error(manifest):
    toml = TOML.replace('shape = ["rows * 4", "blocks"]', 'shape = ["rows - 4", "blocks"]')
    (manifest.directory / "kernel.toml").write_text(toml)
    with pytest.raises(PlanError, match=r"outputs.y.shape: 'rows - 4' = -1, not a positive size"):
        plan_case(load_manifest(manifest.directory), {"rows": 3, "s": 2.0}, F.FP32)


def test_representable_answers_exactly_and_names_the_true_neighbours():
    assert representable(2.0, F.INT8) == (True, ())
    assert representable(-128.0, F.INT8) == (True, ())
    assert representable(1.5, F.INT8) == (False, (1.0, 2.0))
    assert representable(200.0, F.INT8) == (False, (127.0,))
    assert representable(2.0, F.E4) == (True, ()) and representable(-0.0, F.E4) == (True, ())
    assert representable(1.5, F.FP32) == (True, ())
    # 0.1 rounds to float32 0.100000001...: the neighbours straddle 0.1, one of them is that rounding.
    assert representable(0.1, F.FP32) == (False, (0.09999999403953552, 0.10000000149011612))
    assert representable(1e-300, F.FP32) == (False, (0.0, 1.401298464324817e-45))
    assert representable(1e39, F.FP32) == (False, (3.4028234663852886e38,))
    assert representable(float("nan"), F.FP32) == (False, ())
    exact, near = representable(0.1, F.E4)
    assert not exact and len(near) == 2 and near[0] < 0.1 < near[1]
    assert representable(1000.0, F.E4) == (False, (240.0,))  # past E4's largest finite
    assert representable(0.75, F.E5)[0] and representable(7.0, F.INT32) == (True, ())


def test_s_1_5_is_rejected_in_int8_and_2_0_accepted_in_int8_and_e4(manifest):
    with pytest.raises(PlanError, match="nearest: 1.0 and 2.0"):
        plan_case(manifest, {"rows": 1, "s": 1.5}, F.INT8)
    for fmt in (F.INT8, F.E4):
        assert plan_case(manifest, {"rows": 1, "s": 2.0}, fmt).params["s"] == 2.0


@pytest.mark.parametrize(
    ("value", "words"),
    [
        (float("nan"), "s = nan is not finite"),
        (float("inf"), "s = inf is not finite"),
        (10**400, "is too large for a float"),
    ],
)
def test_non_finite_and_huge_floats_are_refused(manifest, value, words):
    with pytest.raises(PlanError, match=words):
        plan_case(manifest, {"rows": 1, "s": value}, F.FP32)


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        ("ceil(k / 128)", 2),
        ("rows * 4", 12),
        ("k // 7 % 5", 3),
        ("max(rows, 2) - min(1, k)", 2),
        ("-rows + 5", 2),
        ("-7 // 2", -4),  # Python's floor semantics
        ("-7 % 3", 2),
        ("ceil(-3 / 2)", -1),
        ("min(rows, k, 2)", 2),
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
        "ceil(rows, 2)",
        "min(rows)",
        "max()",
        "rows ** 2",
        "k / 3",  # not a whole number
        "rows / 0",
        "1.5 * rows",
        "'4'",
        "unknown",
        "rows if rows else 1",
        "[rows]",
        "rows +",
        "1" + "+1" * 5000,  # too deeply nested
    ],
)
def test_anything_else_is_rejected(expression):
    with pytest.raises(ExpressionError):
        evaluate(expression, {"k": 128, "rows": 3})


def test_a_derived_value_that_does_not_evaluate_is_an_error(manifest):
    toml = TOML.replace('blocks = "ceil(k / 128)"', 'blocks = "k / 3"')
    (manifest.directory / "kernel.toml").write_text(toml)
    with pytest.raises(PlanError) as error:
        plan_case(load_manifest(manifest.directory), {"rows": 3, "s": 2.0}, F.FP32)
    assert error.value.key == "derived.blocks"
    assert "'k / 3': 'k / 3' is 128/3, not a whole number" in str(error.value)


def test_check_params_runs_checks_2_and_3_and_names_the_formats_that_accept(manifest):
    check = check_params(manifest, {"rows": 2, "s": 2.0})
    assert dict(check.params) == {"rows": 2, "k": 128, "mode": "a", "s": 2.0}
    assert check.formats == (F.FP32, F.INT8, F.E4)
    assert check_params(manifest, {"rows": 2, "s": 1.5}).formats == (F.FP32, F.E4)
    assert check_params(manifest, {"rows": 2, "s": 0.1}).formats == ()
    with pytest.raises(PlanError, match="rows = 0 is below its min 1"):
        check_params(manifest, {"rows": 0, "s": 1.0})
    toml = TOML.replace('shape = ["rows * 4", "blocks"]', 'shape = ["rows - 4", "blocks"]')
    (manifest.directory / "kernel.toml").write_text(toml)
    with pytest.raises(PlanError, match="not a positive size"):
        check_params(load_manifest(manifest.directory), {"rows": 3, "s": 1.0})


def test_only_entry_points_call_the_validation_seam():
    """No later stage re-validates: plan_case, check_params and representable are
    called only by params.py itself and the entry points (the runner, E06-11; the
    registry, E06-16; :assemble_<name>'s kernel_bin, E06-17)."""
    import ipu_harness

    package = Path(ipu_harness.__file__).parent
    allowed = {"params.py", "runner.py", "registry.py", "kernel_bin.py"}
    calls = re.compile(
        r"\b(plan_case|check_params|representable|_check_(?:params|value|number|format|scalars))\("
    )
    offenders = [
        str(p.relative_to(package))
        for p in package.rglob("*.py")
        if p.name not in allowed and calls.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
