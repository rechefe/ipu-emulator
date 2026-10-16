"""Region placement: declaration order, pins, capacity, rows per format (E06-04)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ipu_harness.manifest import load_manifest
from ipu_harness.params import PlanError, plan_case
from ipu_harness.placement import Placement, default_rows, place
from ipu_isa.formats import ElementFormat as F
from ipu_isa.formats import addressable_rows

SCALE = """\
[kernel]
op = "scale"
formats = ["fp32", "int8", "e4"]

[params]
rows = {{ type = "int", min = 1 }}
s    = {{ type = "float" }}

[inputs.x]
shape = ["rows", 128]
{x_pin}
[outputs.y]
shape = ["rows", 128]
{y_extra}
[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[cases.default]
rows = 4
s = 2.0
"""


def manifest(tmp_path: Path, toml: str, name: str = "scale"):
    directory = tmp_path / name
    directory.mkdir()
    (directory / f"{name}.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(toml)
    return load_manifest(directory)


def bases(placement: Placement) -> dict[str, tuple[int, int]]:
    return {name: (r.base_row, r.rows) for name, r in placement.regions.items()}


def test_scale_with_4_rows_puts_x_at_0_and_y_at_4(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    placement = place(plan_case(m, {"rows": 4, "s": 2.0}, F.FP32))
    assert bases(placement) == {"x": (0, 4), "y": (4, 4)}
    assert placement.rows_used == 8 and placement.region("y").end_row == 8


def test_a_pinned_region_keeps_its_row_and_the_rest_fill_around_it(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="row = 2\n", y_extra=""))
    assert bases(place(plan_case(m, {"rows": 2, "s": 2.0}, F.FP32))) == {"x": (2, 2), "y": (0, 2)}
    # A gap too small for y: y goes after x.
    assert bases(place(plan_case(m, {"rows": 3, "s": 2.0}, F.FP32))) == {"x": (2, 3), "y": (5, 3)}


def test_placement_is_deterministic(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="row = 7\n", y_extra=""))
    plan = plan_case(m, {"rows": 5, "s": 2.0}, F.INT8)
    assert bases(place(plan)) == bases(place(plan))


def test_overlapping_pins_name_both_regions(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="row = 0\n", y_extra="row = 3\n"))
    with pytest.raises(PlanError) as error:
        place(plan_case(m, {"rows": 4, "s": 2.0}, F.FP32))
    assert error.value.key == "outputs.y.row"
    assert "y (rows 3..6) overlaps x (rows 0..3)" in str(error.value)


def test_a_pin_or_regions_past_capacity_are_errors(tmp_path):
    capacity = addressable_rows(F.FP32)
    m = manifest(tmp_path, SCALE.format(x_pin=f"row = {capacity - 2}\n", y_extra=""))
    with pytest.raises(
        PlanError, match=f"x pinned at row {capacity - 2} needs 4 rows, past the {capacity} rows"
    ):
        place(plan_case(m, {"rows": 4, "s": 2.0}, F.FP32))
    m2 = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""), "big")
    with pytest.raises(
        PlanError, match=f"y needs rows {capacity}..{2 * capacity - 1}, past the {capacity} rows"
    ) as error:
        place(plan_case(m2, {"rows": capacity, "s": 2.0}, F.FP32))
    assert error.value.key == "outputs.y.row"


def test_rows_are_the_same_in_every_format_while_capacity_differs(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    fp32 = place(plan_case(m, {"rows": 6, "s": 2.0}, F.FP32))
    e4 = place(plan_case(m, {"rows": 6, "s": 2.0}, F.E4))
    assert bases(fp32) == bases(e4)
    assert e4.capacity == 4 * fp32.capacity


def test_an_accumulator_output_takes_more_rows_in_int8(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra='store = "accumulator"\n'))
    fp32 = place(plan_case(m, {"rows": 2, "s": 2.0}, F.FP32))
    int8 = place(plan_case(m, {"rows": 2, "s": 2.0}, F.INT8))
    assert fp32.region("y").rows == 2 and int8.region("y").rows == 8
    assert int8.region("x").rows == 2


def test_row_counts_from_packing_win_and_must_name_tensors(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    plan = plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)
    assert bases(place(plan, {"x": 9})) == {"x": (0, 9), "y": (9, 4)}
    assert default_rows(plan, m.tensors["x"]) == 4
    with pytest.raises(PlanError, match="row count for 'z', which is no tensor"):
        place(plan, {"z": 1})


PROJECTIONS = [
    f"proj_{kind}_{width}_p4"
    for kind in ("qkv", "outproj", "ffn1", "ffn2")
    for width in (144, 192, 240)
]

PROJECTION_TOML = """\
[kernel]
op = "projection"
formats = ["fp32"]

[params]
k     = {{ type = "int", default = {k} }}
n_out = {{ type = "int", default = {n_out} }}
n_tg  = {{ type = "int", default = {n_tg} }}

[inputs.data]
shape = [4, "k", "n_tg", 128]

[inputs.weights]
shape = ["n_out", "k"]

[outputs.out]
shape = [4, "n_tg", "n_out", 128]

[abi]
data    = "CR2"
weights = "CR3"
out     = "CR4"

[cases.default]
"""


@pytest.mark.parametrize("name", PROJECTIONS)
def test_projection_layouts_come_out_of_declaration_order(tmp_path, name):
    """Guards E07-02: with no pins, placement reproduces ProjectionLayout's bases."""
    app = importlib.import_module(f"ipu_apps.kernels.projections.{name}.app")
    layout = next(
        v.layout
        for v in vars(app).values()
        if isinstance(v, type) and v.__module__ == app.__name__ and hasattr(v, "layout")
    )
    toml = PROJECTION_TOML.format(k=layout.k, n_out=layout.n_out, n_tg=layout.n_tg)
    placement = place(plan_case(manifest(tmp_path, toml, name), {}, F.FP32))
    assert placement.region("data").base_row == layout.data_base_row
    assert placement.region("weights").base_row == layout.weights_base_row
    assert placement.region("out").base_row == layout.output_base_row
