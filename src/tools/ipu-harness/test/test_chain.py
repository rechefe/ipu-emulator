"""Chaining kernels in memory (E06-15): one step's output region is the next
step's input, on one model, with no file between them."""

from __future__ import annotations

import builtins
import io
import os
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.errors import ConfigurationError
from ipu_harness.chain import Chain, Resident
from ipu_harness.manifest import load_manifest
from ipu_harness.params import PlanError
from ipu_harness.runner import RunOptions
from ipu_isa.formats import ElementFormat

#: y = x * s, quantized back to INT8 elements (so y can be the next x). ST.AAQ
#: writes all 512 bytes of POST_AAQ_REG, four INT8 rows, so each store spills
#: into the three rows after its own: the kernels below pin y where the spill
#: lands on nothing a step reads.
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
{x_row}

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


def _kernel(root: Path, name: str = "scale8", *, y_row: str = "", x_row: str = "") -> Path:
    directory = root / "elementwise" / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text(ASM)
    (directory / "reference.py").write_text(REFERENCE)
    (directory / "kernel.toml").write_text(
        TOML.replace('op = "scale8"', f'op = "{name}"').format(y_row=y_row, x_row=x_row)
    )
    return directory


@pytest.fixture
def first(tmp_path):
    """Leaves y at rows 64.. (spilling to 67..)."""
    return load_manifest(_kernel(tmp_path, "first", y_row="row = 64"))


@pytest.fixture
def second(tmp_path):
    """Reads x wherever the chain says, leaves y at rows 128.."""
    return load_manifest(_kernel(tmp_path, "second", y_row="row = 128"))


X = (np.arange(2 * 128).reshape(2, 128) % 17 - 8).astype(np.int8)
SCALE_2 = {"rows": 2, "s": 2.0}
SCALE_3 = {"rows": 2, "s": 3.0}


def test_a_scale_chained_into_a_scale(first, second):
    chain = Chain(ElementFormat.INT8)
    one = chain.run(first, SCALE_2, {"x": X})
    assert one.result.halted and not one.result.comparisons  # not checked
    assert one.resident["y"] == Resident(64, 2, None, ElementFormat.INT8)
    assert chain.model.read_register("lr", 1) == 2  # registers carry over, as on hardware
    two = chain.run(second, SCALE_3, {"x": one.resident["y"]}, check=True)
    assert two.result.placement is not None and two.result.placement.region("x").base_row == 64
    assert two.result.passed, two.result.describe()
    assert two.result.comparisons  # held to the reference of the resident input
    np.testing.assert_array_equal(two.result.outputs["y"], X.astype(np.float64) * 2 * 3)


def test_a_mismatched_layout_fails_before_running(first, second):
    chain = Chain(ElementFormat.INT8)
    resident = chain.run(first, SCALE_2, {"x": X}).resident["y"]
    stats = chain.model.stats()
    for wrong in (
        Resident(resident.base_row, resident.rows, "no_such_pack", resident.format),
        Resident(resident.base_row, resident.rows, resident.pack, ElementFormat.FP32),
        Resident(resident.base_row, resident.rows + 1, resident.pack, resident.format),
    ):
        with pytest.raises(ConfigurationError, match="laid out"):
            chain.run(second, SCALE_3, {"x": wrong})
    assert chain.model.stats() == stats  # nothing ran


def test_a_resident_cannot_move_an_input_pinned_in_kernel_toml(tmp_path, first):
    pinned = load_manifest(_kernel(tmp_path, "pinned", x_row="row = 10", y_row="row = 128"))
    chain = Chain(ElementFormat.INT8)
    resident = chain.run(first, SCALE_2, {"x": X}).resident["y"]
    with pytest.raises(ConfigurationError, match="pinned to row 10"):
        chain.run(pinned, SCALE_3, {"x": resident})


def test_a_resident_overlapping_another_region_is_a_placement_error(tmp_path, first):
    clash = load_manifest(_kernel(tmp_path, "clash", y_row="row = 65"))
    chain = Chain(ElementFormat.INT8)
    resident = chain.run(first, SCALE_2, {"x": X}).resident["y"]  # rows 64..65
    with pytest.raises(PlanError, match="overlaps"):
        chain.run(clash, SCALE_3, {"x": resident})


def test_a_resident_a_later_step_wrote_over_is_stale(tmp_path, first, second):
    again = load_manifest(_kernel(tmp_path, "again", y_row="row = 64"))
    chain = Chain(ElementFormat.INT8)
    one = chain.run(first, SCALE_2, {"x": X}).resident["y"]  # rows 64..65
    two = chain.run(second, SCALE_3, {"x": one}).resident["y"]  # one stays live
    chain.run(again, SCALE_2, {"x": X})  # its y takes rows 64..65 again
    with pytest.raises(ConfigurationError, match="no longer in XMEM"):
        chain.run(second, SCALE_3, {"x": one})
    assert chain.run(first, SCALE_3, {"x": two}).result.halted  # two is still live


def test_a_step_refused_before_it_writes_keeps_the_residents(tmp_path, first, second):
    clash = load_manifest(_kernel(tmp_path, "clash", y_row="row = 65"))
    chain = Chain(ElementFormat.INT8)
    resident = chain.run(first, SCALE_2, {"x": X}).resident["y"]
    with pytest.raises(PlanError, match="overlaps"):
        chain.run(clash, SCALE_3, {"x": resident})
    with pytest.raises(ConfigurationError, match="takes inputs"):
        chain.run(second, SCALE_3, {"xx": resident})
    assert chain.run(second, SCALE_3, {"x": resident}).result.halted


def test_a_resident_from_elsewhere_is_refused(first, second):
    with pytest.raises(ConfigurationError, match="not this chain's"):
        Chain(ElementFormat.INT8).run(
            first, SCALE_2, {"x": Resident(64, 2, None, ElementFormat.INT8)}
        )
    theirs = Chain(ElementFormat.INT8).run(first, SCALE_2, {"x": X}).resident["y"]
    mine = Chain(ElementFormat.INT8)
    mine.run(first, SCALE_2, {"x": X})  # an equal resident, but this chain's own
    with pytest.raises(ConfigurationError, match="not this chain's"):
        mine.run(second, SCALE_3, {"x": theirs})
    with pytest.raises(ConfigurationError, match="rows >= 1"):
        Resident(0, 0, None, ElementFormat.INT8)


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")}


def test_no_tensor_file_is_written_or_read_between_steps(first, second, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chain = Chain(ElementFormat.INT8)
    one = chain.run(first, SCALE_2, {"x": X})
    before = _tree(tmp_path)
    opened = []

    def watching(real):
        def open_(file, mode="r", *args, **kwargs):
            opened.append((str(file), mode))
            return real(file, mode, *args, **kwargs)

        return open_

    monkeypatch.setattr(builtins, "open", watching(builtins.open))
    monkeypatch.setattr(io, "open", watching(io.open))
    monkeypatch.setattr(np, "save", lambda *a, **k: pytest.fail("np.save between steps"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("np.load between steps"))
    chain.run(second, SCALE_3, {"x": one.resident["y"]})
    assert not [f for f, mode in opened if set(mode) & set("wax+") or f.endswith(".npy")]
    assert _tree(tmp_path) == before
    assert os.getcwd() == str(tmp_path)


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        ({}, "takes inputs"),
        ({"x": X, "z": X}, "takes inputs"),
        ({"x": np.full((2, 128), np.nan)}, "not finite"),
        ({"x": X[:1]}, "shape"),
    ],
)
def test_the_inputs_must_be_the_kernels(first, inputs, match):
    with pytest.raises(ConfigurationError, match=match):
        Chain(ElementFormat.INT8).run(first, SCALE_2, inputs)


def test_each_step_writes_its_own_trace(first, second, tmp_path):
    trace = tmp_path / "run.jsonl"
    chain = Chain(ElementFormat.INT8, options=RunOptions(trace=trace))
    one = chain.run(first, SCALE_2, {"x": X})
    chain.run(second, SCALE_3, {"x": one.resident["y"]})
    assert (tmp_path / "run.0.jsonl").is_file() and (tmp_path / "run.1.jsonl").is_file()


def test_a_chain_takes_no_input_files(tmp_path):
    with pytest.raises(ConfigurationError, match="--input"):
        Chain(ElementFormat.INT8, options=RunOptions(inputs={"x": tmp_path / "x.npy"}))


# --- at: a step's own regions where the caller puts them ----------------------------


@pytest.fixture
def free(tmp_path):
    """Places x and y by first fit."""
    return load_manifest(_kernel(tmp_path, "free"))


def test_at_places_an_output_and_an_array_input(free):
    chain = Chain(ElementFormat.INT8)
    step = chain.run(free, SCALE_2, {"x": X}, at={"y": 200, "x": 40})
    placement = step.result.placement
    assert placement is not None
    assert (placement.region("x").base_row, placement.region("y").base_row) == (40, 200)
    assert step.resident["y"] == Resident(200, 2, None, ElementFormat.INT8)
    np.testing.assert_array_equal(
        chain.run(free, SCALE_3, {"x": step.resident["y"]}, check=True).result.outputs["y"],
        X.astype(np.float64) * 6,
    )


def test_a_region_placed_with_at_makes_what_it_covers_stale(first, free, second):
    chain = Chain(ElementFormat.INT8)
    one = chain.run(first, SCALE_2, {"x": X}).resident["y"]  # rows 64..65
    chain.run(free, SCALE_2, {"x": X}, at={"y": 65})
    with pytest.raises(ConfigurationError, match="no longer in XMEM"):
        chain.run(second, SCALE_3, {"x": one})


@pytest.mark.parametrize(
    ("at", "match"),
    [
        ({"z": 0}, "at names 'z', which is no tensor"),
        ({"x": 0}, "at names input x, which is resident"),
        ({"y": 100}, "pinned to row 128 in kernel.toml; at gives 100"),
        ({"y": -1}, "not a row"),
    ],
)
def test_at_must_name_a_region_the_step_places(first, second, at, match):
    chain = Chain(ElementFormat.INT8)
    resident = chain.run(first, SCALE_2, {"x": X}).resident["y"]
    with pytest.raises(ConfigurationError, match=match):
        chain.run(second, SCALE_3, {"x": resident}, at=at)
    assert chain.run(second, SCALE_3, {"x": resident}, at={"y": 128}).result.halted


def test_an_aaq_spill_makes_the_rows_after_an_output_stale(first, free, second):
    chain = Chain(ElementFormat.INT8)
    one = chain.run(first, SCALE_2, {"x": X}).resident["y"]  # rows 64..65
    chain.run(free, SCALE_2, {"x": X}, at={"y": 62})  # y 62..63, its ST.AAQ spill 64..66
    with pytest.raises(ConfigurationError, match="no longer in XMEM"):
        chain.run(second, SCALE_3, {"x": one})


def test_a_resident_meets_an_expression_pin(tmp_path, first):
    pinned = load_manifest(_kernel(tmp_path, "expr", x_row='row = "rows * 32"', y_row="row = 128"))
    chain = Chain(ElementFormat.INT8)
    resident = chain.run(first, SCALE_2, {"x": X}).resident["y"]  # row 64 == 2 * 32
    assert chain.run(pinned, SCALE_3, {"x": resident}).result.halted


def _maybe(tmp_path: Path) -> Path:
    """free, with an output z only when ``extra``."""
    directory = _kernel(tmp_path, "maybe")
    toml = (directory / "kernel.toml").read_text()
    toml = toml.replace("s    = {", 'extra = { type = "bool", default = false }\ns    = {')
    toml = toml.replace("[abi]", '[outputs.z]\nshape = ["rows", 128]\nwhen = "extra"\n\n[abi]')
    toml = toml.replace('s    = "CR5"', 's    = "CR5"\nz    = "CR6"')
    (directory / "kernel.toml").write_text(toml)
    return directory


def test_an_output_its_case_lacks_is_no_resident(tmp_path):
    step = Chain(ElementFormat.INT8).run(load_manifest(_maybe(tmp_path)), SCALE_2, {"x": X})
    assert set(step.resident) == {"y"}


def test_at_cannot_name_an_output_its_case_lacks(tmp_path):
    with pytest.raises(ConfigurationError, match="'z', which is no tensor of this case"):
        Chain(ElementFormat.INT8).run(
            load_manifest(_maybe(tmp_path)), SCALE_2, {"x": X}, at={"z": 0}
        )


# --- hold: a resident later steps place around ------------------------------------


def test_a_held_resident_is_placed_around(free, second):
    chain = Chain(ElementFormat.INT8)
    held = chain.hold(chain.run(free, SCALE_2, {"x": X}, at={"y": 0, "x": 10}).resident["y"])
    step = chain.run(free, SCALE_3, {"x": X})  # first fit would put x on rows 0..1
    assert step.result.placement is not None
    assert step.result.placement.region("x").base_row >= 2
    two = chain.run(second, SCALE_3, {"x": held}, check=True)  # still live; read in place
    np.testing.assert_array_equal(two.result.outputs["y"], X.astype(np.float64) * 6)


def test_first_fit_leaves_room_for_a_spill_below_a_held_resident(free, second):
    chain = Chain(ElementFormat.INT8)
    held = chain.hold(chain.run(free, SCALE_2, {"x": X}, at={"y": 4, "x": 10}).resident["y"])
    step = chain.run(free, SCALE_3, {"x": X})  # y at 2..3 would spill onto rows 4..6
    assert step.result.placement is not None
    assert all(
        r.end_row + 3 <= 4 or r.base_row >= 6 for r in step.result.placement.regions.values()
    )
    two = chain.run(second, SCALE_3, {"x": held}, check=True)
    np.testing.assert_array_equal(two.result.outputs["y"], X.astype(np.float64) * 6)


@pytest.mark.parametrize("at", [{"y": 64}, {"y": 62}])  # on it; spilling onto it
def test_a_step_that_would_write_a_held_resident_fails_before_it_runs(first, free, at):
    chain = Chain(ElementFormat.INT8)
    held = chain.hold(chain.run(first, SCALE_2, {"x": X}).resident["y"])  # rows 64..65
    stats = chain.model.stats()
    with pytest.raises(PlanError, match=r"held tensor \(rows 64..65\)"):
        chain.run(free, SCALE_2, {"x": X}, at=at)
    assert chain.model.stats() == stats
    chain.release(held)
    chain.run(free, SCALE_2, {"x": X}, at={"y": 64})
    with pytest.raises(ConfigurationError, match="cannot hold"):
        chain.hold(held)  # released, then written over


def test_only_a_live_resident_of_the_chain_can_be_held_or_released(first):
    chain = Chain(ElementFormat.INT8)
    resident = chain.run(first, SCALE_2, {"x": X}).resident["y"]
    with pytest.raises(ConfigurationError, match="cannot hold"):
        chain.hold(Resident(64, 2, None, ElementFormat.INT8))
    with pytest.raises(ConfigurationError, match="does not hold"):
        chain.release(resident)
    assert chain.hold(chain.hold(resident)) is resident
    chain.release(resident)
