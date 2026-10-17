"""Pack transforms (E06-05): a named transform on an input or an output, and a
region's stored rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_harness.manifest import ManifestError, load_manifest
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
