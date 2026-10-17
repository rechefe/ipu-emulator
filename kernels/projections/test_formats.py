"""The projections in more than one element format (E07-03, Q44, Q50).

The 8 ``proj_p4`` entries run in FP32 and INT8 (``bazel test
//kernels/projections:<entry>`` runs each case in both); the 4 two-group L3
entries run in FP32 only. These tests pin what those runs cannot show:

* the INT8 inputs' ``range`` keeps the outputs off the INT8 rails, so an INT8
  run that passes is not one where every output saturated (a K = 192 dot
  product of full-range INT8 values saturates almost all of them);
* why no template runs in E4: ``ACTIVATE.QUANTIZE`` refuses an 8-bit float
  format (the AaQ placeholder, OQ-20; Q25, Q35);
* why the two-group templates do not run in INT8: ``ST.AAQ`` stores four rows
  in an 8-bit format (OQ-35), and their store order zeroes rows already stored;
* a pack transform laid out at FP32's element size fails INT8 with a size
  error, not a wrong answer.

When OQ-20 or OQ-35 is answered, its test fails: add the format to the
templates' ``formats`` and delete the test.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.errors import EmulatorError
from ipu_harness.cases import expand, inputs_for
from ipu_harness.family import load_kernel, read_registry
from ipu_harness.manifest import load_manifest
from ipu_harness.params import PlanError, plan_case
from ipu_harness.reference import evaluate, load_reference
from ipu_harness.runner import run_kernel
from ipu_isa.formats import ElementFormat as F

FAMILY = Path(__file__).parent
REGISTRY = FAMILY / "registry.toml"
#: The entries that run in INT8: those over proj_p4.
INT8_ENTRIES = [e.name for e in read_registry(REGISTRY) if e.template == "proj_p4"]

#: The most of an entry's INT8 outputs that may round onto -128 or 127.
#: Measured with range = [-2, 2] (E07-03), from K = 192 to 480: qkv_192 0.002%,
#: outproj_192 0%, ffn1_192 0.001%, ffn2_192 0.157%, qkv_240 0.004%,
#: outproj_240 0.007%, ffn1_240 0.003%, ffn2_240 0.358%.
MAX_SATURATED = 0.01


def _int8_reference(name: str) -> np.ndarray:
    manifest = load_kernel(f"{REGISTRY}::{name}")
    case = next(c for c in expand(manifest) if c.name == "default")
    plan = plan_case(manifest, case.values, F.INT8, case=case.decl)
    inputs = inputs_for(case, plan).tensors
    reference = load_reference(manifest)
    assert reference is not None
    return evaluate(reference, inputs, plan.params)["y"]


@pytest.mark.parametrize("name", INT8_ENTRIES)
def test_int8_inputs_keep_the_outputs_off_the_rails(name):
    y = np.rint(_int8_reference(name))
    saturated = float(np.mean((y <= -128) | (y >= 127)))
    print(f"{name}: {saturated:.4%} of y saturates in INT8")  # the record E07-03 keeps
    assert saturated < MAX_SATURATED, f"{name}: {saturated:.2%} of y saturates in INT8"
    # Nor so narrow a range that the outputs are small integers a wrong sum could still hit.
    assert np.mean(np.abs(y) >= 16) > 0.25, f"{name}: y is mostly within 16 of zero"


def _template_copy(tmp_path: Path, template: str, formats: str | None = None) -> Path:
    """``template`` in a copy of the family folder, so a test can edit its files;
    with ``formats``, its ``[kernel] formats`` replaced."""
    copy = tmp_path / "projections"
    shutil.copytree(FAMILY, copy, ignore=shutil.ignore_patterns("test_*.py", "BUILD.bazel"))
    folder = copy / template
    if formats is not None:
        toml = folder / "kernel.toml"
        text, n = re.subn(
            r"^formats = .*$", f"formats = {formats}", toml.read_text(), count=1, flags=re.M
        )
        assert n == 1
        toml.write_text(text)
    return folder


def _small_case(manifest):
    """The template's default case: its own small shape, run in a second."""
    return next(c for c in expand(manifest) if c.name == "default")


@pytest.mark.parametrize(
    ("template", "formats"),
    [
        ("proj_p4", '["fp32", "int8", "e4"]'),
        ("proj_p4_2tg", '["fp32", "e4"]'),
        ("proj_p4_2tg_silu", '["fp32", "e4"]'),
    ],
)
def test_e4_waits_for_the_aaq_stage(tmp_path, template, formats):
    manifest = load_manifest(_template_copy(tmp_path, template, formats))
    with pytest.raises(EmulatorError, match="ACTIVATE.QUANTIZE instruction requires INT8 mode"):
        run_kernel(manifest, _small_case(manifest), F.E4)


@pytest.mark.parametrize("template", ["proj_p4_2tg", "proj_p4_2tg_silu"])
def test_two_group_int8_waits_for_st_aaq_to_store_one_row(tmp_path, template):
    manifest = load_manifest(_template_copy(tmp_path, template, '["fp32", "int8"]'))
    result = run_kernel(manifest, _small_case(manifest), F.INT8)
    y = result.outputs["y"]
    # Group 0's last three stores spill over group 1's first three rows of each stream.
    assert np.all(y[:, 1, :3] == 0) and not result.passed, result.describe()


#: channel_major at FP32's element size: each element as its four bytes'
#: worth of lanes, so an INT8 region gets four times the rows it was placed in.
FP32_SIZED_PACK = """\
import numpy as np

from ipu_kernels.pack import channel_major as _right


class channel_major:
    rows = staticmethod(_right.rows)

    @staticmethod
    def to_rows(a, plan):
        return np.repeat(_right.to_rows(a, plan), 4, axis=-1).reshape(-1, 128)

    from_rows = staticmethod(_right.from_rows)
"""


def test_a_pack_at_fp32_element_size_fails_int8_with_a_size_error(tmp_path):
    template = _template_copy(tmp_path, "proj_p4")
    (template / "pack.py").write_text(FP32_SIZED_PACK)
    manifest = load_manifest(template)
    with pytest.raises(PlanError, match=r"inputs\.b\.pack.*rows"):
        run_kernel(manifest, _small_case(manifest), F.INT8)
