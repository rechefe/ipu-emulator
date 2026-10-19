"""Conditional tensors and bindings: ``when`` (E06 gap G6).

The pointwise conv twins differ only by a bias region and by what CR10
holds (the bias base, or the tail size): one kernel.toml with a ``bias``
parameter says both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_harness.abi import register_writes
from ipu_harness.conditions import Condition
from ipu_harness.manifest import KernelManifest, ManifestError, load_manifest
from ipu_harness.params import plan_case
from ipu_harness.placement import place
from ipu_isa.formats import ElementFormat as F

TOML = """\
[kernel]
op = "conv"
formats = ["fp32"]

[params]
rows = { type = "int", min = 1 }
bias = { type = "bool", default = false }

[derived]
tail = "rows - 1"

[inputs.x]
shape = ["rows", 128]

[inputs.b]
shape = [1, 128]
when = "bias"

[outputs.y]
shape = ["rows", 128]

[abi]
x    = "CR2"
y    = "CR3"
b    = "CR10"
tail = { register = "CR10", when = "not bias" }

[cases.default]
rows = 2
"""


def manifest(tmp_path: Path, toml: str = TOML) -> KernelManifest:
    directory = tmp_path / "conv"
    directory.mkdir()
    (directory / "conv.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(toml)
    return load_manifest(directory)


def test_a_conditional_tensor_and_binding_load(tmp_path):
    m = manifest(tmp_path)
    assert m.tensors["b"].when == Condition.of("bias")
    assert {n: str(c) for n, c in m.abi_when.items()} == {"b": "bias", "tail": "not bias"}


def test_with_bias_the_bias_region_is_placed_and_bound(tmp_path):
    plan = plan_case(manifest(tmp_path), {"rows": 2, "bias": True}, F.FP32)
    assert set(plan.manifest.tensors) == {"x", "b", "y"}
    placement = place(plan)
    writes = {w.register: w.source for w in register_writes(plan, placement)}
    assert writes["CR10"].startswith("b (region base")
    assert placement.region("b").base_row == 2


def test_without_bias_the_region_is_gone_and_cr10_holds_the_tail(tmp_path):
    plan = plan_case(manifest(tmp_path), {"rows": 2}, F.FP32)
    assert set(plan.manifest.tensors) == {"x", "y"}
    writes = {w.register: w.value for w in register_writes(plan, place(plan))}
    assert writes["CR10"] == 1  # tail = rows - 1
    assert place(plan).region("y").base_row == 2


def test_the_narrowed_view_is_shared_by_plans_with_the_same_answer(tmp_path):
    m = manifest(tmp_path)
    a = plan_case(m, {"rows": 2, "bias": True}, F.FP32).manifest
    b = plan_case(m, {"rows": 5, "bias": True}, F.FP32).manifest
    assert a is b
    assert plan_case(m, {"rows": 2}, F.FP32).manifest is not a


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('when = "bias"', 'when = "rows"', "must be bool parameters"),
        ('when = "bias"', 'when = "bias or rows"', "must be bool parameters"),
        ('when = "bias"', 'when = "bias and rows"', "must be bool parameters"),
        ('when = "not bias"', 'when = "bias"', "tail and b both bind CR10"),
        ('b    = "CR10"', 'b    = { register = "CR10", when = "not bias" }', "takes its tensor's"),
        ('x    = "CR2"', 'x    = { register = "CR2", when = "bias" }', r"condition \(always\)"),
    ],
)
def test_a_bad_condition_is_refused(tmp_path, old, new, message):
    with pytest.raises(ManifestError, match=message):
        manifest(tmp_path, TOML.replace(old, new, 1))


def test_a_region_after_an_absent_tensor_keeps_its_place(tmp_path):
    toml = TOML.replace(
        '[outputs.y]\nshape = ["rows", 128]', '[outputs.y]\nshape = ["rows", 128]\nafter = "b"'
    )
    m = manifest(tmp_path, toml)
    without = place(plan_case(m, {"rows": 2}, F.FP32))
    assert without.region("y").base_row == 2  # b is absent: y keeps its place
    with_bias = place(plan_case(m, {"rows": 2, "bias": True}, F.FP32))
    assert (with_bias.region("b").base_row, with_bias.region("y").base_row) == (2, 3)


PLACED = TOML.replace(
    'tail = "rows - 1"', 'tail = "rows - 1"\npb = "base(b) + 1"\npy = "base(y) + 1"'
)


def test_a_placed_value_reading_an_absent_tensor_does_not_stop_the_others(tmp_path):
    bound = 'pb   = { register = "CR6", when = "bias" }\npy   = "CR7"\ntail = { register'
    m = manifest(tmp_path, PLACED.replace("tail = { register", bound))
    # x at rows 0..1; with bias, b at row 2 and y at 3, else y at 2.
    for bias, cr6, cr7 in ((False, None, 3), (True, 3, 4)):
        plan = plan_case(m, {"rows": 2, "bias": bias}, F.FP32)
        got = {w.register: w.value for w in register_writes(plan, place(plan))}
        assert (got.get("CR6"), got["CR7"]) == (cr6, cr7)


def test_a_placed_value_reading_a_conditional_tensor_takes_its_condition(tmp_path):
    toml = PLACED.replace("tail = { register", 'pb   = "CR6"\ntail = { register')
    with pytest.raises(ManifestError, match="bind it with when = 'bias'"):
        manifest(tmp_path, toml)


@pytest.mark.parametrize(
    ("text", "values", "holds"),
    [
        ("bias or relu", {"bias": False, "relu": True}, True),
        ("bias or relu", {"bias": False, "relu": False}, False),
        ("not bias or relu", {"bias": True, "relu": False}, False),
        ("not bias or relu", {"bias": False, "relu": False}, True),
    ],
)
def test_a_condition_joined_by_or_holds_when_one_term_does(text, values, holds):
    assert Condition.parse(text).holds(values) is holds


def test_or_conditions_compare_by_their_terms_and_exclude_only_when_every_pair_does():
    assert Condition.parse("relu   or  bias") == Condition.parse("bias or relu")
    assert str(Condition.parse("relu or not bias")) == "not bias or relu"
    bias, no_bias = Condition.of("bias"), Condition.of("bias", negated=True)
    assert bias.excludes(no_bias) and no_bias.excludes(bias)
    assert not Condition.parse("bias or relu").excludes(no_bias)  # relu and not bias may hold
    assert not bias.excludes(None)


def test_a_tensor_exists_when_either_term_holds(tmp_path):
    toml = TOML.replace(
        'bias = { type = "bool", default = false }',
        'bias = { type = "bool", default = false }\nrelu = { type = "bool", default = false }',
    )
    toml = toml.replace('when = "bias"', 'when = "bias or relu"', 1)
    toml = toml.replace('b    = "CR10"', 'b    = "CR11"')
    m = manifest(tmp_path, toml)
    for bias, relu, present in ((False, False, False), (False, True, True), (True, False, True)):
        plan = plan_case(m, {"rows": 2, "bias": bias, "relu": relu}, F.FP32)
        assert ("b" in plan.manifest.tensors) is present


@pytest.mark.parametrize(
    ("text", "values", "holds"),
    [
        ("not bias and not relu", {"bias": False, "relu": False}, True),
        ("not bias and not relu", {"bias": True, "relu": False}, False),
        ("not bias and not relu", {"bias": False, "relu": True}, False),
        # and binds tighter than or: (bias and relu) or skip
        ("bias and relu or skip", {"bias": True, "relu": False, "skip": False}, False),
        ("bias and relu or skip", {"bias": False, "relu": False, "skip": True}, True),
        ("bias and relu or skip", {"bias": True, "relu": True, "skip": False}, True),
    ],
)
def test_a_condition_joined_by_and_holds_when_every_term_does(text, values, holds):
    assert Condition.parse(text).holds(values) is holds


def test_and_conditions_compare_by_their_clauses_and_exclude_their_complement():
    assert Condition.parse("relu  and  not bias") == Condition.parse("not bias and relu")
    assert str(Condition.parse("skip or relu and not bias")) == "not bias and relu or skip"
    neither = Condition.parse("not bias and not relu")
    either = Condition.parse("bias or relu")
    assert neither.excludes(either) and either.excludes(neither)
    assert not Condition.parse("not bias and relu").excludes(Condition.of("bias", negated=True))
    assert neither.params == ("bias", "relu")


def test_a_binding_under_and_shares_a_register_with_its_complement(tmp_path):
    # The pointwise conv's CR10 (E07-20): the bias base when seeded
    # (bias or relu), the tail size when neither is set.
    toml = TOML.replace(
        'bias = { type = "bool", default = false }',
        'bias = { type = "bool", default = false }\nrelu = { type = "bool", default = false }',
    )
    toml = toml.replace('when = "bias"', 'when = "bias or relu"', 1)
    toml = toml.replace('when = "not bias"', 'when = "not bias and not relu"')
    m = manifest(tmp_path, toml)
    for bias, relu, cr10 in ((False, False, 1), (False, True, 2), (True, False, 2)):
        plan = plan_case(m, {"rows": 2, "bias": bias, "relu": relu}, F.FP32)
        writes = {w.register: w.value for w in register_writes(plan, place(plan))}
        assert writes["CR10"] == cr10  # tail = rows - 1, or b's base row
    (tmp_path / "overlap").mkdir()
    with pytest.raises(ManifestError, match="tail and b both bind CR10"):
        manifest(tmp_path / "overlap", toml.replace("not bias and not relu", "not bias"))
