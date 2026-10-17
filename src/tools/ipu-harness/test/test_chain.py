"""Chaining kernels in memory (E06-15): one step's output region is the next
step's input, on one model, with no file between them."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.errors import ConfigurationError
from ipu_harness.chain import Chain, Resident
from ipu_harness.manifest import load_manifest
from ipu_harness.params import PlanError
from ipu_isa.formats import ElementFormat

#: y = x * s, quantized back to INT8 elements (so y can be the next x).
ASM = """\
    SET LR0 CR0 ;
    SET LR1 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity CR15 ;;
    ST.AAQ LR1 {{ cr.y }} ;;
    ADD LR1 LR1 CR1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
"""

TOML = """\
[kernel]
op = "scale8"
formats = ["int8"]

[params]
rows = {{ type = "int", min = 1, max = 8 }}
s    = {{ type = "float" }}

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]
{y_row}
[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[golden]
reference = "reference.py:reference"
tolerance = {{ int8 = "EXACT" }}

[cases.default]
rows = 2
s = 2.0
"""

REFERENCE = '''\
import numpy as np


def reference(x, *, rows, s):
    """y = x * s, as INT8 holds it."""
    return {"y": np.clip(np.rint(np.asarray(x, dtype=np.float64) * s), -128, 127)}
'''


def _kernel(root: Path, *, y_row: str = "") -> Path:
    directory = root / "elementwise" / "scale8"
    directory.mkdir(parents=True)
    (directory / "scale8.asm").write_text(ASM)
    (directory / "reference.py").write_text(REFERENCE)
    (directory / "kernel.toml").write_text(TOML.format(y_row=y_row))
    return directory


@pytest.fixture
def scale8(tmp_path):
    return load_manifest(_kernel(tmp_path))


X = (np.arange(2 * 128).reshape(2, 128) % 17 - 8).astype(np.int8)


def test_a_scale_chained_into_a_scale(scale8):
    chain = Chain(ElementFormat.INT8)
    first = chain.run(scale8, {"rows": 2, "s": 2.0}, {"x": X})
    assert first.result.halted
    second = chain.run(scale8, {"rows": 2, "s": 3.0}, {"x": first.resident["y"]}, compare=True)
    assert second.result.passed, second.result.describe()
    assert second.result.comparisons  # held to the reference of the resident input
    np.testing.assert_array_equal(second.result.outputs["y"], X.astype(np.float64) * 2 * 3)


def test_a_mismatched_layout_fails_before_running(scale8):
    chain = Chain(ElementFormat.INT8)
    first = chain.run(scale8, {"rows": 2, "s": 2.0}, {"x": X})
    resident = first.resident["y"]
    stats = chain.model.stats()
    for wrong in (
        Resident(resident.base_row, resident.rows, "rows_to_lanes", resident.format),
        Resident(resident.base_row, resident.rows, resident.pack, ElementFormat.FP32),
        Resident(resident.base_row, resident.rows + 1, resident.pack, resident.format),
    ):
        with pytest.raises(ConfigurationError, match="laid out"):
            chain.run(scale8, {"rows": 2, "s": 3.0}, {"x": wrong})
    assert chain.model.stats() == stats  # nothing ran


def test_a_resident_overlapping_another_region_is_a_placement_error(tmp_path):
    pinned = load_manifest(_kernel(tmp_path, y_row="row = 0"))
    chain = Chain(ElementFormat.INT8)
    resident = Resident(1, 2, None, ElementFormat.INT8)
    with pytest.raises(PlanError, match="overlaps"):
        chain.run(pinned, {"rows": 2, "s": 3.0}, {"x": resident})


def test_no_file_is_read_or_written_between_steps(scale8, monkeypatch):
    chain = Chain(ElementFormat.INT8)
    first = chain.run(scale8, {"rows": 2, "s": 2.0}, {"x": X})
    opened = []
    real_open = builtins.open

    def watching(file, mode="r", *args, **kwargs):
        opened.append((str(file), mode))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", watching)
    monkeypatch.setattr(np, "save", lambda *a, **k: pytest.fail("np.save between steps"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("np.load between steps"))
    chain.run(scale8, {"rows": 2, "s": 3.0}, {"x": first.resident["y"]})
    assert not [f for f, mode in opened if "w" in mode or "a" in mode or f.endswith(".npy")]


def test_the_inputs_must_be_the_kernels(scale8):
    with pytest.raises(ConfigurationError, match="takes inputs"):
        Chain(ElementFormat.INT8).run(scale8, {"rows": 2, "s": 2.0}, {})
