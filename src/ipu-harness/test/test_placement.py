"""Region placement: declaration order, pins, capacity, rows per format (E06-04)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import MappingProxyType

import pytest

from ipu_harness.cases import expand
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.params import PlanError, plan_case
from ipu_harness.placement import Placement, default_rows, place
from ipu_harness.runner import stage
from ipu_harness.variants import variant
from ipu_isa.formats import ElementFormat as F
from ipu_isa.formats import addressable_rows

SCALE = """\
[kernel]
op = "scale"
formats = ["fp32", "e3", "e4"]

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


def manifest(tmp_path: Path, toml: str, name: str = "scale") -> KernelManifest:
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
    toml = SCALE.format(x_pin="row = 7\n", y_extra="")
    first = plan_case(manifest(tmp_path, toml, "a"), {"rows": 5, "s": 2.0}, variant("e3"))
    second = plan_case(manifest(tmp_path, toml, "b"), {"rows": 5, "s": 2.0}, variant("e3"))
    assert bases(place(first)) == bases(place(second)) == {"x": (7, 5), "y": (0, 5)}


def test_a_free_region_fills_the_gap_between_two_pins(tmp_path):
    toml = SCALE.format(
        x_pin="row = 6\n", y_extra="row = 0\n[scratch.t]\nshape = [3, 128]\n"
    ).replace('s    = "CR5"', 's    = "CR5"\nt    = "CR6"')
    # y at 0..1 and x at 6..7 leave rows 2..5: the 3-row scratch fits there, not after x.
    assert bases(place(plan_case(manifest(tmp_path, toml), {"rows": 2, "s": 2.0}, F.FP32))) == {
        "x": (6, 2),
        "y": (0, 2),
        "t": (2, 3),
    }


def test_scratch_is_placed_after_inputs_and_outputs(tmp_path):
    toml = SCALE.format(x_pin="", y_extra="[scratch.t]\nshape = [3, 128]\nfill = 0.0\n").replace(
        's    = "CR5"', 's    = "CR5"\nt    = "CR6"'
    )
    m = manifest(tmp_path, toml)
    placement = place(plan_case(m, {"rows": 4, "s": 2.0}, F.FP32))
    assert bases(placement) == {"x": (0, 4), "y": (4, 4), "t": (8, 3)}
    assert placement.region("t").role == "scratch"


def test_a_scratch_region_after_the_input_comes_before_the_output(tmp_path):
    scratch = '[scratch.t]\nshape = [3, 128]\nafter = "x"\n'
    toml = SCALE.format(x_pin="", y_extra=scratch).replace(
        's    = "CR5"', 's    = "CR5"\nt    = "CR6"'
    )
    m = manifest(tmp_path, toml)
    assert m.tensors["t"].after == "x"
    placement = place(plan_case(m, {"rows": 4, "s": 2.0}, F.FP32))
    assert bases(placement) == {"x": (0, 4), "y": (7, 4), "t": (4, 3)}


def test_a_pin_can_be_an_expression_of_the_parameters(tmp_path):
    toml = SCALE.format(x_pin='row = "rows * 2"', y_extra="")
    m = manifest(tmp_path, toml)
    assert m.tensors["x"].row == "rows * 2"
    plan = plan_case(m, {"rows": 3, "s": 2.0}, F.FP32)
    assert plan.pins == {"x": 6}
    assert bases(place(plan)) == {"x": (6, 3), "y": (0, 3)}


def test_overlapping_pins_name_both_regions(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="row = 0\n", y_extra="row = 3\n"))
    with pytest.raises(PlanError) as error:
        place(plan_case(m, {"rows": 4, "s": 2.0}, F.FP32))
    assert error.value.key == "outputs.y.row"
    assert "y (rows 3..6) overlaps x (rows 0..3)" in str(error.value)


def test_a_callers_pin_places_its_tensor_and_the_rest_fill_around_it(tmp_path):
    """A chained input already in XMEM (E06-15)."""
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    plan = plan_case(m, {"rows": 2, "s": 2.0}, F.FP32)
    assert bases(place(plan, pins={"x": 3})) == {"x": (3, 2), "y": (0, 2)}
    with pytest.raises(ValueError, match="no tensors"):
        place(plan, pins={"z": 0})


def test_a_callers_pin_overlapping_a_row_pin_is_an_error(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra="row = 0\n"))
    with pytest.raises(PlanError, match="overlaps"):
        place(plan_case(m, {"rows": 2, "s": 2.0}, F.FP32), pins={"x": 1})


def test_reserved_rows_are_placed_around_and_a_pin_on_them_is_an_error(tmp_path):
    """Rows a chain holds (E06-15) are no region's."""
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    plan = plan_case(m, {"rows": 2, "s": 2.0}, F.FP32)
    placement = place(plan, reserved={"held": (1, 2)})
    assert bases(placement) == {"x": (3, 2), "y": (5, 2)}
    assert [(r.name, r.base_row, r.rows) for r in placement.reserved] == [("held", 1, 2)]
    with pytest.raises(PlanError, match=r"x \(rows 2..3\) overlaps held \(rows 1..2\)"):
        place(plan, pins={"x": 2}, reserved={"held": (1, 2)})


def test_avoided_rows_are_left_by_first_fit_but_may_be_pinned(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    plan = plan_case(m, {"rows": 2, "s": 2.0}, F.FP32)
    assert bases(place(plan, avoid=[(0, 3)])) == {"x": (3, 2), "y": (5, 2)}
    assert bases(place(plan, pins={"x": 1}, avoid=[(0, 3)])) == {"x": (1, 2), "y": (3, 2)}


def test_an_aaq_store_writes_only_its_row_so_the_next_region_is_untouched(tmp_path):
    """ADR 0007: ACTIVATE.QUANTIZE stores exactly one row (ST.AAQ's 512-byte
    spill past an output left with it), so x may follow y directly."""
    toml = SCALE.format(x_pin="", y_extra="row = 0\n").replace(
        "[abi]\n", '[derived]\nqf = "case_qf"\n\n[abi]\nqf   = "CR6"\n'
    )
    m = manifest(tmp_path, toml)
    (m.directory / "scale.asm").write_text(
        "ACTIVATE.QUANTIZE identity CR15.L {{ cr.qf }}.L {{ cr.y }} LR0 ;;\nHALT ;;\n"
    )
    case = next(c for c in expand(m) if c.name == "default")
    staged = stage(m, case, variant("e4"))
    start, size = staged.region_bytes("x")
    assert staged.placement.region("y").end_row == staged.placement.region("x").base_row
    before = staged.model.read_xmem(start, size)
    staged.model.run()
    assert staged.model.read_xmem(start, size) == before


CAPACITY = addressable_rows(F.FP32)


def test_a_pin_past_capacity_is_an_error(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin=f"row = {CAPACITY - 2}\n", y_extra=""))
    with pytest.raises(
        PlanError, match=f"x pinned at row {CAPACITY - 2} needs 4 rows, past the {CAPACITY} rows"
    ) as error:
        place(plan_case(m, {"rows": 4, "s": 2.0}, F.FP32))
    assert error.value.key == "inputs.x.row"


def test_regions_that_do_not_fit_are_an_error(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    with pytest.raises(
        PlanError, match=f"y needs rows {CAPACITY}..{2 * CAPACITY - 1}, past the {CAPACITY} rows"
    ) as error:
        place(plan_case(m, {"rows": CAPACITY, "s": 2.0}, F.FP32))
    assert error.value.key == "outputs.y.row"


def test_a_pin_at_the_end_of_xmem_pushes_a_free_region_past_capacity(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=f"row = {CAPACITY - 4}\n"))
    plan = plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)
    assert bases(place(plan, {"x": CAPACITY - 4})) == {
        "x": (0, CAPACITY - 4),
        "y": (CAPACITY - 4, 4),
    }
    with pytest.raises(PlanError, match=f"x needs rows {CAPACITY}..") as error:
        place(plan, {"x": CAPACITY - 3})
    assert error.value.key == "inputs.x.row"


def test_rows_are_the_same_in_every_format_while_capacity_differs(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    fp32 = place(plan_case(m, {"rows": 6, "s": 2.0}, F.FP32))
    e4 = place(plan_case(m, {"rows": 6, "s": 2.0}, variant("e4")))
    assert bases(fp32) == bases(e4)
    assert e4.capacity == 4 * fp32.capacity


def test_row_counts_from_packing_win_and_must_name_tensors(tmp_path):
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    plan = plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)
    assert bases(place(plan, {"x": 9})) == {"x": (0, 9), "y": (9, 4)}
    assert default_rows(plan, m.tensors["x"]) == 4
    with pytest.raises(ValueError, match="row count for 'z', which is no tensor"):
        place(plan, {"z": 1})
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match=f"row count {bad!r} for 'x'"):
            place(plan, {"x": bad})


def test_place_takes_the_plan_as_checked(tmp_path):
    """The one seam is plan_case (E06-03): place re-checks no parameter, so a plan
    whose values plan_case would refuse still places. (test_params.py's source
    scan keeps the checks out of this module.)"""
    m = manifest(tmp_path, SCALE.format(x_pin="", y_extra=""))
    plan = plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)
    unchecked = dataclasses.replace(
        plan,
        params=MappingProxyType({"rows": -1, "s": "?"}),
        shapes=MappingProxyType({"x": (2, 0), "y": (4, 128)}),  # plan_case refuses a 0 dim
    )
    assert bases(place(unchecked)) == {"x": (0, 0), "y": (0, 4)}


#: Each projection's (k, n_out, n_tg) and its data, weights and output base rows
#: under the old ProjectionLayout (E02 baselines; frozen in E07-02, which
#: deleted it).
PROJECTIONS = {
    "proj_qkv_144_p4": (144, 432, 2, (0, 1152, 2016)),
    "proj_outproj_144_p4": (144, 144, 2, (0, 1152, 1440)),
    "proj_ffn1_144_p4": (144, 288, 2, (0, 1152, 1728)),
    "proj_ffn2_144_p4": (288, 144, 2, (0, 2304, 2736)),
    "proj_qkv_192_p4": (192, 576, 1, (0, 768, 1920)),
    "proj_outproj_192_p4": (192, 192, 1, (0, 768, 1152)),
    "proj_ffn1_192_p4": (192, 384, 1, (0, 768, 1536)),
    "proj_ffn2_192_p4": (384, 192, 1, (0, 1536, 2112)),
    "proj_qkv_240_p4": (240, 720, 1, (0, 960, 2400)),
    "proj_outproj_240_p4": (240, 240, 1, (0, 960, 1440)),
    "proj_ffn1_240_p4": (240, 480, 1, (0, 960, 1920)),
    "proj_ffn2_240_p4": (480, 240, 1, (0, 1920, 2880)),
}

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


@pytest.mark.parametrize("name", sorted(PROJECTIONS))
def test_projection_layouts_come_out_of_declaration_order(tmp_path, name):
    """Guards E07-02: with no pins, placement reproduces ProjectionLayout's bases.

    The shapes are the old XMEM layout with default_rows, not the logical
    shapes; ipu-kernels' test_pack places the logical shapes with packing's
    row counts.
    """
    k, n_out, n_tg, bases = PROJECTIONS[name]
    toml = PROJECTION_TOML.format(k=k, n_out=n_out, n_tg=n_tg)
    placement = place(plan_case(manifest(tmp_path, toml, name), {}, F.FP32))
    assert tuple(placement.region(t).base_row for t in ("data", "weights", "out")) == bases
