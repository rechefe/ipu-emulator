"""Pack transforms (E06-05): a named transform on an input or an output, and a
region's stored rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_harness.manifest import load_manifest
from ipu_harness.packing import pack, stored_rows, unpack
from ipu_harness.params import plan_case
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
