"""Conditional tensors and bindings: ``when`` (E06 gap G6).

The pointwise conv twins differ only by a bias region and by what CR10
holds (the bias base, or the tail size): one kernel.toml with a ``bias``
parameter says both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_harness.abi import register_writes
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
    assert m.tensors["b"].when == "bias"
    assert dict(m.abi_when) == {"b": "bias", "tail": "not bias"}


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
        ('when = "bias"', 'when = "rows"', "must name a bool parameter"),
        ('when = "not bias"', 'when = "bias"', "tail and b both bind CR10"),
        ('b    = "CR10"', 'b    = { register = "CR10", when = "not bias" }', "takes its tensor's"),
    ],
)
def test_a_bad_condition_is_refused(tmp_path, old, new, message):
    with pytest.raises(ManifestError, match=message):
        manifest(tmp_path, TOML.replace(old, new, 1))
