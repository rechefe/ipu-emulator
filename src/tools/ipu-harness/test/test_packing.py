"""Pack transforms (E06-05): a named transform on an input or an output, and a
region's stored rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_harness.manifest import KernelManifest, ManifestError, load_manifest
from ipu_harness.packing import pack, stored_rows, unpack
from ipu_harness.params import PlanError, plan_case
from ipu_isa.formats import ElementFormat as F

TOML = """\
[kernel]
op = "copy"
formats = ["fp32"]

[params]
rows = { type = "int", min = 1 }

[inputs.x]
shape = ["rows", 128]
pack = "{x_pack}"

[outputs.y]
shape = ["rows", 128]
pack = "{y_pack}"

[abi]
x = "CR2"
y = "CR3"
rows = "CR4"

[cases.default]
rows = 2
"""

# Rows stored last first, with one spare row after them.
PACK_PY = """\
import numpy as np


class reversed_rows:
    @staticmethod
    def rows(shape):
        return shape[0] + 1

    @staticmethod
    def to_rows(a, plan):
        return np.vstack([a[::-1], np.zeros((1, 128))])

    @staticmethod
    def from_rows(r, plan, shape):
        return r[: shape[0]][::-1]
"""


def manifest(tmp_path: Path, x_pack: str, y_pack: str):
    directory = tmp_path / "copy"
    directory.mkdir()
    (directory / "copy.asm").write_text("HALT ;;\n")
    (directory / "pack.py").write_text(PACK_PY)
    (directory / "kernel.toml").write_text(
        TOML.replace("{x_pack}", x_pack).replace("{y_pack}", y_pack)
    )
    return load_manifest(directory)


def test_an_output_names_a_pack_transform_and_unpack_uses_it(tmp_path):
    m = manifest(tmp_path, "reversed_rows", "reversed_rows")
    assert m.tensors["y"].pack == "reversed_rows"
    plan = plan_case(m, {"rows": 2}, F.FP32)
    y = m.tensors["y"]
    assert stored_rows(y, plan) == 3
    values = np.arange(256, dtype=np.float64).reshape(2, 128)
    raw = pack(y, values, plan)
    stored = np.frombuffer(raw, dtype="<f4").reshape(3, 128)
    np.testing.assert_array_equal(stored[0], values[1])  # stored by the output's transform
    np.testing.assert_array_equal(unpack(y, raw, plan), values)


def _family_kernel(tmp_path: Path, *, own: str | None) -> KernelManifest:
    """``copy`` in a family folder with a shared pack.py, and its own if given."""
    directory = tmp_path / "fam" / "copy"
    directory.mkdir(parents=True)
    (directory / "copy.asm").write_text("HALT ;;\n")
    (directory.parent / "pack.py").write_text(PACK_PY)  # reversed_rows, one spare row
    if own is not None:
        (directory / "pack.py").write_text(own)
    toml = TOML.replace("{x_pack}", "reversed_rows").replace("{y_pack}", "reversed_rows")
    (directory / "kernel.toml").write_text(toml)
    return load_manifest(directory)


def test_a_familys_pack_py_serves_its_kernels(tmp_path):
    m = _family_kernel(tmp_path, own=None)
    plan = plan_case(m, {"rows": 2}, F.FP32)
    assert stored_rows(m.tensors["y"], plan) == 3  # the family's reversed_rows


def test_the_family_folders_pack_py_comes_before_the_familys_package(tmp_path, monkeypatch):
    package = tmp_path / "fakepkg"
    (package / "fam").mkdir(parents=True)
    for folder in (package, package / "fam"):
        (folder / "__init__.py").write_text("")
    (package / "fam" / "pack.py").write_text(PACK_PY.replace("shape[0] + 1", "shape[0] + 9"))
    monkeypatch.syspath_prepend(str(tmp_path))
    m = _family_kernel(tmp_path, own=None)
    plan = plan_case(m, {"rows": 2}, F.FP32)
    assert stored_rows(m.tensors["y"], plan, package="fakepkg") == 3  # the folder's, not +9


def test_the_kernels_own_pack_py_wins_over_its_familys(tmp_path):
    own = PACK_PY.replace("return shape[0] + 1", "return shape[0] + 5")
    m = _family_kernel(tmp_path, own=own)
    plan = plan_case(m, {"rows": 2}, F.FP32)
    assert stored_rows(m.tensors["y"], plan) == 7


def test_a_pack_no_file_defines_names_where_it_looked(tmp_path):
    m = _family_kernel(tmp_path, own=None)
    (m.directory.parent / "pack.py").write_text("")
    plan = plan_case(m, {"rows": 2}, F.FP32)
    with pytest.raises(PlanError, match="the family's fam/pack.py"):
        stored_rows(m.tensors["y"], plan)


#: Rows that depend on the plan, not only the shape (a tiled layout's padding).
PLANNED_PY = PACK_PY.replace(
    "    @staticmethod\n    def rows(shape):\n        return shape[0] + 1",
    "    @staticmethod\n    def rows(shape, plan):\n        return shape[0] + plan.params['rows']",
)


def test_a_transforms_rows_may_read_the_plan_and_one_that_does_not_still_works(tmp_path):
    planned = _family_kernel(tmp_path / "planned", own=PLANNED_PY)
    plan = plan_case(planned, {"rows": 3}, F.FP32)
    assert stored_rows(planned.tensors["y"], plan) == 6  # 3 rows + rows = 3
    shape_only = _family_kernel(tmp_path / "shape_only", own=None)  # rows(shape)
    assert stored_rows(shape_only.tensors["y"], plan_case(shape_only, {"rows": 3}, F.FP32)) == 4


#: The forms a transform's rows may take, each adding a distinct count.
FORMS_PY = """\
import numpy as np


class _Layout:
    def to_rows(self, a, plan):
        return np.vstack([a, np.zeros((self.extra, 128))])

    def from_rows(self, r, plan, shape):
        return r[: shape[0]]


class _Bound(_Layout):
    extra = 0

    def rows(self, shape):
        return shape[0] + 2


class _KeywordPlan(_Layout):
    def rows(self, shape, *, plan):
        return shape[0] + 10 * plan.params["rows"]


class _Variadic(_Layout):
    def rows(self, *args):
        return args[0][0] + (100 if len(args) == 2 else 0)


bound = _Bound()
keyword_plan = _KeywordPlan()
variadic = _Variadic()
"""


@pytest.mark.parametrize(
    ("name", "rows"),
    [("bound", 2 + 2), ("keyword_plan", 2 + 20), ("variadic", 2 + 100)],
)
def test_each_rows_form_gets_the_plan_it_asks_for(tmp_path, name, rows):
    m = _family_kernel(tmp_path, own=FORMS_PY)
    toml = (m.directory / "kernel.toml").read_text().replace("reversed_rows", name)
    (m.directory / "kernel.toml").write_text(toml)
    m = load_manifest(m.directory)
    assert stored_rows(m.tensors["y"], plan_case(m, {"rows": 2}, F.FP32)) == rows


# --- a scratch region's starting bytes (E06 gap G2) ----------------------------------------

SCRATCH_TOML = """\
[kernel]
op = "masked"
formats = ["fp32", "int8"]

[params]
rows = { type = "int", min = 1 }

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[scratch.mask]
shape = [1, 128]
{content}

[abi]
x = "CR2"
y = "CR3"
mask = "CR4"

[cases.default]
rows = 2
"""

MASK_PY = """\
def mask(plan):
    return bytes([0xFF, 0x00] * (plan.manifest.directory.name == "masked") * 64) * (
        4 if plan.format.value == "fp32" else 1
    )


def short(plan):
    return b"\\xff"
"""


def scratch_kernel(tmp_path: Path, content: str):
    directory = tmp_path / "masked"
    directory.mkdir()
    (directory / "masked.asm").write_text("HALT ;;\n")
    (directory / "mask.py").write_text(MASK_PY)
    (directory / "kernel.toml").write_text(SCRATCH_TOML.replace("{content}", content))
    return directory


def test_a_filled_region_is_its_fill_in_every_element(tmp_path):
    from ipu_harness.packing import region_fill

    m = load_manifest(scratch_kernel(tmp_path, "fill = 1.5"))
    plan = plan_case(m, {"rows": 2}, F.FP32)
    assert region_fill(m.tensors["mask"], plan, 2) == np.full(256, 1.5, "<f4").tobytes()


def test_a_fill_may_be_an_exact_expression_of_the_case(tmp_path):
    """layernorm's 1/n and -1/n rows: evaluated exactly, rounded when stored."""
    from ipu_harness.packing import region_fill

    m = load_manifest(scratch_kernel(tmp_path, 'fill = "-1 / (rows + 1)"'))
    assert m.tensors["mask"].fill == "-1 / (rows + 1)"
    fp32 = plan_case(m, {"rows": 2}, F.FP32)
    assert region_fill(m.tensors["mask"], fp32, 1) == np.full(128, -1 / 3, "<f4").tobytes()
    int8 = plan_case(m, {"rows": 2}, F.INT8)  # -1/3 rounds to 0 in INT8
    assert region_fill(m.tensors["mask"], int8, 1) == bytes(128)


@pytest.mark.parametrize(
    ("fill", "match"),
    [('"1 / rowz"', "names no parameter"), ('"base(x)"', "names no parameter")],
)
def test_a_fill_expression_reads_only_the_cases_values(tmp_path, fill, match):
    with pytest.raises(ManifestError, match=match):
        load_manifest(scratch_kernel(tmp_path, f"fill = {fill}"))


def test_a_constant_input_range_is_allowed(tmp_path):
    from ipu_harness.cases import expand, inputs_for

    directory = scratch_kernel(tmp_path, "")
    toml = directory / "kernel.toml"
    toml.write_text(
        toml.read_text().replace(
            '[inputs.x]\nshape = ["rows", 128]', '[inputs.x]\nshape = ["rows", 128]\nrange = [0, 0]'
        )
    )
    m = load_manifest(directory)
    case = expand(m)[0]
    for fmt in (F.FP32, F.INT8):
        x = inputs_for(case, plan_case(m, case.values, fmt, case=case.decl)).tensors["x"]
        assert not x.any()


def test_a_bytes_region_is_its_functions_bytes(tmp_path):
    import dataclasses

    from ipu_harness.packing import region_fill

    m = load_manifest(scratch_kernel(tmp_path, ""))
    mask = dataclasses.replace(m.tensors["mask"], content="mask.py:mask")
    plan = plan_case(m, {"rows": 2}, F.FP32)
    assert region_fill(mask, plan, 1) == bytes([0xFF, 0x00] * 64) * 4
    with pytest.raises(PlanError, match="returned 1 bytes; the region holds 512"):
        region_fill(dataclasses.replace(mask, content="mask.py:short"), plan, 1)


def test_a_bytes_region_loads(tmp_path):
    directory = scratch_kernel(tmp_path, 'bytes = "mask.py:mask"')
    assert load_manifest(directory).tensors["mask"].content == "mask.py:mask"


def test_fill_and_bytes_together_are_refused(tmp_path):
    directory = scratch_kernel(tmp_path, 'fill = 1.0\nbytes = "mask.py:mask"')
    with pytest.raises(ManifestError, match="takes 'fill' or 'bytes', not both"):
        load_manifest(directory)
