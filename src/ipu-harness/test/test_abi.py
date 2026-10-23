"""Registers from [abi]: bases, integers, element scalars, the CR-read check (E06-06)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_state import IpuState
from ipu_emu.reference import ReferenceModel
from ipu_harness import manifest as manifest_module
from ipu_harness.abi import apply, check_cr_reads, register_writes, template_names
from ipu_harness.manifest import KernelManifest, ManifestError, load_manifest
from ipu_harness.packing import stored_rows
from ipu_harness.params import PlanError, plan_case
from ipu_harness.placement import place
from ipu_harness.variants import Variant, variant
from ipu_isa.cr_halves import DEFAULT_DS
from ipu_isa.formats import ElementFormat

SCALE = """\
[kernel]
op = "scale"
formats = ["fp32", "e3", "e4"]

[params]
rows = {{ type = "int", min = 1{unit} }}
s    = {{ type = "float" }}

[derived]
neg = "-rows"

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"
neg  = "LR3"

[cases.default]
rows = 4
s = 2.0
"""


def kernel(
    tmp_path: Path, toml: str, family: str = "elementwise", name: str = "scale"
) -> KernelManifest:
    directory = tmp_path / family / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(toml)
    return load_manifest(directory)


FP32, E3, E4 = variant("fp32"), variant("e3"), variant("e4")


def writes(m: KernelManifest, params: dict[str, object], fmt: Variant) -> dict[str, int]:
    plan = plan_case(m, params, fmt)
    return {w.register: w.value for w in register_writes(plan, place(plan))}


#: Hand-encoded 2.0: FP32 0 10000000 000...; NX8, the MULT constant layout
#: (ADR 0007): element code 0 (exponent 0, mantissa 0: 2^(scale - 127)) at
#: scale 128, then fe (3 or 4) at bits 19:16 and sign 1 at bits 23:20.
TWO = {FP32: 0x40000000, E3: 0x138000, E4: 0x148000}


@pytest.mark.parametrize("fmt", [FP32, E3, E4])
def test_scale_bases_and_integers_are_the_same_in_every_format_and_s_is_encoded(tmp_path, fmt):
    m = kernel(tmp_path, SCALE.format(unit=""))
    got = writes(m, {"rows": 4, "s": 2.0}, fmt)
    assert {k: got[k] for k in ("CR2", "CR3", "CR4")} == {"CR2": 0, "CR3": 4, "CR4": 4}
    assert got["CR5"] == TWO[fmt]
    assert got["LR3"] == 0xFFFFFFFC  # -4, two's complement in 32 bits
    # The default DS in CR15.L: valid_elements 128 in bits 7:0, one partition
    # (code 0 at bits 9:8), valid_partitions 1 at bits 15:12.
    assert got["CR15"] == DEFAULT_DS == 0x1080


def test_a_negative_nx8_scalar_is_a_signed_constant(tmp_path):
    m = kernel(tmp_path, SCALE.format(unit=""))
    # -3.0 = -1.5 x 2^1: code 1 000 100 (sign, exponent 0, mantissa 4) at scale 128.
    assert writes(m, {"rows": 4, "s": -3.0}, E4)["CR5"] == 0x148084


def test_a_tensor_in_a_bytes_kernel_is_its_base_in_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest_module, "BYTE_UNIT_KERNELS", frozenset({"legacy/scale"}))
    m = kernel(tmp_path, SCALE.format(unit=', unit = "bytes"'), family="legacy")
    got = writes(m, {"rows": 4, "s": 2.0}, FP32)
    assert (got["CR2"], got["CR3"]) == (0, 4 * 512)  # base row 4, 512 B a FP32 row
    assert writes(m, {"rows": 4, "s": 2.0}, E4)["CR3"] == 4 * 128


def test_a_tensor_binding_with_an_offset_is_its_base_plus_the_offset(tmp_path):
    toml = SCALE.format(unit="").replace('y    = "CR3"', 'y    = { register = "CR3", offset = -1 }')
    m = kernel(tmp_path, toml)
    assert m.abi["y"] == "CR3" and m.abi_offsets["y"] == -1
    assert writes(m, {"rows": 4, "s": 2.0}, FP32)["CR3"] == 3  # y at row 4, minus 1


def test_a_stage_table_binds_a_region_at_its_base_whatever_the_kernels_offset(tmp_path):
    toml = SCALE.format(unit="").replace('y    = "CR3"', 'y    = { register = "CR3", offset = -1 }')
    m = kernel(tmp_path, toml)
    plan = plan_case(m, {"rows": 4, "s": 2.0}, FP32)
    got = {w.register: w.value for w in register_writes(plan, place(plan), abi={"y": "CR6"})}
    assert got["CR6"] == 4  # y's base row: the [abi] offset is the kernel's binding only


def test_a_derived_value_may_read_a_regions_base(tmp_path):
    """Several offsets of one region (the x128 matmuls' weights + 1..3, E07)."""
    toml = SCALE.format(unit="").replace(
        'neg = "-rows"', 'neg = "-rows"\ny1 = "base(y) + 1"\ny2 = "y1 + rows"'
    )
    toml = toml.replace('neg  = "LR3"', 'neg  = "LR3"\ny1   = "CR6"\ny2   = "CR7"')
    m = kernel(tmp_path, toml)
    assert m.placed == {"y1": frozenset({"y"}), "y2": frozenset({"y"})}
    plan = plan_case(m, {"rows": 4, "s": 2.0}, FP32)
    assert "y1" not in plan.derived and plan.derived["neg"] == -4
    got = writes(m, {"rows": 4, "s": 2.0}, FP32)
    assert (got["CR6"], got["CR7"]) == (5, 9)  # y at row 4: 4 + 1, then 5 + 4


@pytest.mark.parametrize(("fmt", "row", "stride"), [(FP32, 256, 1), (E4, 1024, 4)])
def test_a_pin_and_a_placed_value_may_read_the_formats_row_bytes(tmp_path, fmt, row, stride):
    """A legacy layout at a fixed byte address in every format (fully_connected, E07-21)."""
    toml = SCALE.format(unit="").replace(
        'neg = "-rows"', 'neg = "-rows"\nstride = "512 / row_bytes"'
    )
    toml = toml.replace('neg  = "LR3"', 'neg  = "LR3"\nstride = "CR6"')
    y = '[outputs.y]\nshape = ["rows", 128]'
    toml = toml.replace(y, f'{y}\nrow = "131072 / row_bytes"')
    m = kernel(tmp_path, toml)
    assert m.placed == {"stride": frozenset()}
    plan = plan_case(m, {"rows": 4, "s": 2.0}, fmt)
    assert plan.pins["y"] == row and "stride" not in plan.derived
    got = writes(m, {"rows": 4, "s": 2.0}, fmt)
    assert (got["CR3"], got["CR6"]) == (row, stride)


@pytest.mark.parametrize(
    ("derived", "where", "match"),
    [
        ('row_bytes = "4"', "", "not be cr, lr, format, shapes, row_bytes"),
        ("", 'shape = ["row_bytes", 128]', "names no parameter or derived value: ['row_bytes']"),
        (
            'w = "row_bytes"',
            'shape = ["w", 128]',
            "reads ['w'], which read a region's base or row_bytes",
        ),
    ],
)
def test_only_a_pin_and_abi_read_row_bytes(tmp_path, derived, where, match):
    toml = SCALE.format(unit="").replace('neg = "-rows"', f'neg = "-rows"\n{derived}')
    if where:
        toml = toml.replace('[outputs.y]\nshape = ["rows", 128]', f"[outputs.y]\n{where}")
    with pytest.raises(ManifestError, match=re.escape(match)):
        kernel(tmp_path, toml)


def test_a_region_may_reach_the_program_only_through_a_derived_value(tmp_path):
    """A region bound at a parameter-dependent offset (sample_descriptors'
    per-cell table, E07-15) needs no binding of its own; one that no bound
    value reads still does."""
    toml = SCALE.format(unit="").replace('neg = "-rows"', 'neg = "-rows"\nyr = "base(y) + rows"')
    bound = toml.replace('y    = "CR3"', 'yr   = "CR3"')
    got = writes(kernel(tmp_path, bound), {"rows": 4, "s": 2.0}, FP32)
    assert got["CR3"] == 8  # y at row 4, plus rows
    with pytest.raises(ManifestError, match="does not bind tensor 'y'"):
        kernel(tmp_path / "unbound", toml.replace('y    = "CR3"\n', ""))
    # Bound only in some cases: y, which always exists, would lose its base.
    flagged = toml.replace("[params]\n", '[params]\nflag = { type = "bool", default = true }\n')
    flagged = flagged.replace('y    = "CR3"', 'yr   = { register = "CR3", when = "flag" }')
    with pytest.raises(ManifestError, match="does not bind tensor 'y'"):
        kernel(tmp_path / "conditional", flagged)


def test_a_bytes_kernels_base_counts_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest_module, "BYTE_UNIT_KERNELS", frozenset({"elementwise/scale"}))
    toml = SCALE.format(unit=', unit = "bytes"').replace(
        'neg = "-rows"', 'neg = "-rows"\nyb = "base(y) + 1"'
    )
    m = kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nyb   = "CR6"'))
    assert writes(m, {"rows": 4, "s": 2.0}, FP32)["CR6"] == 4 * 512 + 1


@pytest.mark.parametrize(
    ("derived", "where", "match"),
    [
        ('w = "base(z) + 1"', "", "base(z) names no tensor"),
        ('w = "base(1)"', "", "base() takes one tensor name"),
        ('w = "base(y) + w"', "", "names no parameter or derived value: ['w']"),
        ('w = "base(y)"', 'shape = ["w", 128]', "reads ['w'], which read a region's base"),
        (
            'w = "base(x)"',
            'shape = ["rows", 128]\nrow = "w + 4"',
            "only [abi] may read such a value",
        ),
    ],
)
def test_a_placed_value_needs_a_tensor_and_only_abi_reads_it(tmp_path, derived, where, match):
    toml = SCALE.format(unit="").replace('neg = "-rows"', f'neg = "-rows"\n{derived}')
    if where:
        toml = toml.replace('[outputs.y]\nshape = ["rows", 128]', f"[outputs.y]\n{where}")
    with pytest.raises(ManifestError, match=re.escape(match)):
        kernel(tmp_path, toml)


def test_a_placed_value_cannot_be_the_cost(tmp_path):
    toml = SCALE.format(unit="").replace(
        'formats = ["fp32", "e3", "e4"]', 'formats = ["fp32", "e3", "e4"]\ncost = "w"'
    )
    with pytest.raises(ManifestError, match=r"\[kernel\] cost 'w' reads \['w'\]"):
        kernel(tmp_path, toml.replace('neg = "-rows"', 'neg = "-rows"\nw = "base(y)"'))


def test_a_stage_abi_may_bind_a_placed_value(tmp_path):
    toml = SCALE.format(unit="").replace('neg = "-rows"', 'neg = "-rows"\ny1 = "base(y) + 1"')
    m = kernel(tmp_path, toml)
    plan = plan_case(m, {"rows": 4, "s": 2.0}, FP32)
    got = {w.register: w.value for w in register_writes(plan, place(plan), abi={"y1": "CR8"})}
    assert got["CR8"] == 5


def test_a_derived_ds_binds_any_cr(tmp_path):
    """ds(valid, partitions): the DS half, without its bit layout in the file."""
    toml = SCALE.format(unit="").replace(
        'neg = "-rows"', 'neg = "-rows"\ndtok = "ds(rows * 16, 2)"'
    )
    m = kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\ndtok = "CR13"'))
    got = writes(m, {"rows": 4, "s": 2.0}, FP32)
    # valid elements, partitions 2 (code 1 at bits 9:8), valid_partitions 2 (bits 15:12).
    assert got["CR13"] == 64 | 1 << 8 | 2 << 12


@pytest.mark.parametrize(
    ("expression", "word"),
    [
        ("ds(128)", "128 + 4096"),
        ("ds(64, partitions=2)", "64 + 256 + 2 * 4096"),
        ("ds(16, 8, 4)", "16 + 3 * 256 + 4 * 4096"),
        ("ds(16, mask_type=1)", "16 + 1 * 1024 + 4096"),  # the mask type from bit 10
        ("ds(0)", "4096"),
        ("qf(1, 4)", "8 + 4 * 16 + 256"),  # num_bits, fe, sign
        ("qf(0, 3, bits=6)", "6 + 3 * 16"),
        ("qf(1, 0, 8, dynamic=1)", "8 + 256 + 4096"),
        ("constant(132, 128, 4)", "132 + 128 * 256 + 4 * 65536 + 1048576"),
        ("halves(4224, 328)", "4224 + 328 * 65536"),
        ("halves(ds(128), qf(1, 4))", "4224 + 328 * 65536"),
    ],
)
def test_a_cr_helper_takes_its_later_arguments_by_position_or_name(tmp_path, expression, word):
    toml = SCALE.format(unit="").replace(
        'neg = "-rows"', f'neg = "-rows"\nd = "{expression}"\ne = "{word}"'
    )
    toml = toml.replace('neg  = "LR3"', 'neg  = "LR3"\nd    = "CR13"\ne    = "CR14"')
    got = writes(kernel(tmp_path, toml), {"rows": 4, "s": 2.0}, FP32)
    assert got["CR13"] == got["CR14"]


def test_case_qf_is_the_variants_qf_half(tmp_path):
    toml = SCALE.format(unit="").replace('neg = "-rows"', 'neg = "-rows"\nq = "case_qf"')
    m = kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nq    = "CR13"'))
    assert m.placed == {"q": frozenset()}
    assert writes(m, {"rows": 4, "s": 2.0}, E4)["CR13"] == 0x148
    assert writes(m, {"rows": 4, "s": 2.0}, E3)["CR13"] == 0x138
    assert writes(m, {"rows": 4, "s": 2.0}, FP32)["CR13"] == 0  # FP32 ignores the QF


@pytest.mark.parametrize(
    ("expression", "match"),
    [
        ("ds(64, 3)", "num_partitions must be 1, 2, 4 or 8"),
        ("ds(64, 1, 0)", "valid_partitions must be 1..num_partitions"),
        ("ds(64, 2, 2, 3)", "mask_type NaN"),  # NX8 has no NaN (OQ-37)
        ("ds(rows / 3)", "whole numbers"),
        ("ds(129)", "valid_elements must be 0..128"),
        ("ds(-1)", "valid_elements must be 0..128"),
        ("qf(1, 8)", "exp_format"),
        ("constant(256, 127, 4)", "does not fit"),
        ("halves(65536)", "is not 16 bits"),
    ],
)
def test_a_bad_cr_helper_value_is_located(tmp_path, expression, match):
    toml = SCALE.format(unit="").replace('neg = "-rows"', f'neg = "-rows"\nd = "{expression}"')
    m = kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nd    = "CR13"'))
    with pytest.raises(PlanError, match=re.escape(match)):
        plan_case(m, {"rows": 4, "s": 2.0}, FP32)


@pytest.mark.parametrize(
    ("expression", "match"),
    [
        ("ds(64, 1, 1, 0, 1)", r"ds\(\) takes"),
        ("ds()", r"ds\(\) takes"),
        ("ds(64, valid=8)", r"ds\(\) takes"),
        ("ds(64, 8, partitions=2)", r"ds\(\) got"),
        ("qf(1)", r"qf\(\) takes"),
        ("dstructure(128)", r"dstructure\(\) was replaced by ds\(\)"),
    ],
)
def test_a_misshapen_cr_helper_call_is_a_load_error(tmp_path, expression, match):
    toml = SCALE.format(unit="").replace('neg = "-rows"', f'neg = "-rows"\nd = "{expression}"')
    with pytest.raises(ManifestError, match=match):
        kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nd    = "CR13"'))


def test_a_tensor_pinned_at_row_0_can_be_read_through_cr0(tmp_path):
    toml = SCALE.format(unit="").replace(
        '[inputs.x]\nshape = ["rows", 128]', '[inputs.x]\nshape = ["rows", 128]\nrow = 0'
    )
    toml = toml.replace('x    = "CR2"', 'x    = "CR0"')
    m = kernel(tmp_path, toml)
    got = writes(m, {"rows": 4, "s": 2.0}, FP32)
    assert "CR0" not in got and "CR2" not in got  # nothing is written for x
    assert template_names(m)["cr"]["x"] == "CR0"


def test_binding_cr15_replaces_the_default_ds(tmp_path):
    m = kernel(tmp_path, SCALE.format(unit="").replace('s    = "CR5"', 's    = "CR15"'))
    got = writes(m, {"rows": 4, "s": 2.0}, FP32)
    assert got["CR15"] == 0x40000000 and "CR5" not in got


def test_a_str_parameter_cannot_be_bound(tmp_path):
    toml = SCALE.format(unit="").replace(
        's    = { type = "float" }',
        's    = { type = "float" }\nmode = { type = "str", default = "a" }',
    )
    with pytest.raises(ManifestError, match="mode is a str parameter"):
        kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nmode = "CR6"'))


def test_template_names_and_apply_through_the_register_file(tmp_path):
    m = kernel(tmp_path, SCALE.format(unit=""))
    names = template_names(m)
    assert dict(names["cr"]) == {"x": "CR2", "y": "CR3", "rows": "CR4", "s": "CR5"}
    assert dict(names["lr"]) == {"neg": "LR3"}
    plan = plan_case(m, {"rows": 4, "s": 2.0}, FP32)
    model = ReferenceModel(ElementFormat.FP32)
    apply(model, register_writes(plan, place(plan)))
    assert [model.read_register("cr", i) for i in (2, 3, 4, 5)] == [0, 4, 4, 0x40000000]
    assert model.read_register("lr", 3) == 0xFFFFFFFC


# --- the CR-read check -----------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "    SET lr1 cr9;;",  # a CR operand
        "    ADD lr0 lr1 cr9;;",  # an LR-or-CR operand spelled CRn
        "    BZ cr9 top;;",  # a pseudo-instruction (BEQ cr9 CR0 top)
        "    SET lr1 CR9;;",  # any case
        "    SET lr0 cr2; SET lr1 cr9;;",  # not the word's first instruction
        "    AGG.SUM lr0 cr9.l;;",  # a half-CR operand (ADR 0007)
        "    SET lr1 cr9.H;;",  # a CR half read by an LR instruction
    ],
)
def test_a_read_of_an_unbound_cr_names_its_line(tmp_path, line):
    m = kernel(tmp_path, SCALE.format(unit=""))
    program = f"top:\n    SET lr0 cr2;;\n{line}\n    B top;;\n"
    message = rf"scale.asm, rendered line 3 \({re.escape(repr(line.strip()))}\): reads CR9"
    with pytest.raises(ConfigurationError, match=message):
        check_cr_reads(m, program)


def test_hardwired_bound_and_default_ds_crs_and_labels_named_like_crs_pass(tmp_path):
    m = kernel(tmp_path, SCALE.format(unit=""))
    program = (
        "cr9:\n"  # a label, not a register
        "    SET lr0 cr0;;\n    SET lr1 cr1;;\n    SET lr2 cr5;;\n"
        "    ADD lr0 lr1 lr2;;\n"  # an LR-or-CR operand spelled LRn
        "    MULT.VE lr0 cr2 0 lr0 cr15.l cr15.h;;\n"
        "    B cr9;;\n"
    )
    check_cr_reads(m, program)


def test_a_value_that_does_not_fit_a_register_is_located(tmp_path):
    toml = SCALE.format(unit="").replace(
        'neg = "-rows"', 'neg = "-rows"\nhuge = "rows * 1000000000"'
    )
    m = kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nhuge = "LR4"'))
    assert writes(m, {"rows": 4, "s": 2.0}, FP32)["LR4"] == 4_000_000_000  # an unsigned word
    plan = plan_case(m, {"rows": 5, "s": 2.0}, FP32)
    with pytest.raises(PlanError, match="does not fit a 32-bit register") as error:
        register_writes(plan, place(plan))
    assert error.value.key == "abi.huge"


# --- one register for two pinned regions (vector_tour's acc and relu, E07-26) -------------

#: ``y2`` is stored right after ``y``'s 4 rows; the program reaches it from CR3.
TWO_OUTPUTS = (
    SCALE.format(unit="")
    .replace(
        '[outputs.y]\nshape = ["rows", 128]\n',
        "[outputs.y]\nshape = [4, 128]\nrow = 8\n\n[outputs.y2]\nshape = [4, 128]\nrow = ROW\n",
    )
    .replace('y    = "CR3"', 'y    = "CR3"\ny2   = { register = "CR3", offset = -4 }')
)


def test_two_pinned_regions_share_a_register_that_reaches_both(tmp_path):
    m = kernel(tmp_path, TWO_OUTPUTS.replace("ROW", "12"))
    assert writes(m, {"rows": 4, "s": 2.0}, FP32)["CR3"] == 8


@pytest.mark.parametrize(
    "row",
    [pytest.param("13", id="row-plus-offset-9-not-8"), pytest.param('"rows + 8"', id="expression")],
)
def test_regions_whose_pins_disagree_do_not_share_a_register(tmp_path, row):
    with pytest.raises(ManifestError, match="y2 and y both bind CR3"):
        kernel(tmp_path, TWO_OUTPUTS.replace("ROW", row))


def test_a_shared_register_a_caller_pin_splits_is_refused(tmp_path):
    m = kernel(tmp_path, TWO_OUTPUTS.replace("ROW", "12"))
    plan = plan_case(m, {"rows": 4, "s": 2.0}, FP32)
    with pytest.raises(PlanError, match="y2 .region base -4. = 8, but y .region base. put 16") as e:
        register_writes(plan, place(plan, pins={"y": 16}))  # a chain places y elsewhere
    assert e.value.key == "abi.y2"


# --- the projections' 13 CRs (guards E07-02) ----------------------------------------------

PROJECTION = """\
[kernel]
op = "projection"
formats = ["fp32"]

[params]
k     = {{ type = "int", default = {k} }}
n_out = {{ type = "int", default = {n_out} }}
n_tg  = {{ type = "int", default = {n_tg} }}
n_tok = {{ type = "int", default = {n_tok} }}

[derived]
neg_one         = "-1"
full_bound      = "126"
w_stride        = "ceil(k / 128)"
chunk_count     = "ceil(k / 128)"
last_chunk      = "ceil(k / 128) - 1"
tail_bound      = "k - 128 * (ceil(k / 128) - 1) - 2"
data_stream_str = "k * n_tg"
out_stream_str  = "n_tg * n_out"
stream_count    = "4"

[inputs.data]
shape = [4, "n_tg", "k", "n_tok"]
pack = "channel_major"

[inputs.weights]
shape = ["n_out", "k"]
pack = "output_major_weights"

[outputs.out]
shape = [4, "n_tg", "n_out", 128]

[abi]
data            = "CR2"
weights         = "CR3"
out             = "CR4"
neg_one         = "CR5"
full_bound      = "CR6"
tail_bound      = "CR7"
w_stride        = "CR8"
n_out           = "CR9"
chunk_count     = "CR10"
last_chunk      = "CR11"
data_stream_str = "CR12"
out_stream_str  = "CR13"
stream_count    = "CR14"

[cases.default]
"""

#: Each projection's (k, n_out, n_tok, n_tg) and the CR2..CR15 the old harness's
#: setup() wrote (E02 baselines; frozen in E07-02, which deleted it).
PROJECTIONS = {
    "proj_qkv_144_p4": (144, 432, 128, 2, (0, 1152, 2016, -1, 126, 14, 2, 432, 2, 1, 288, 864, 4)),
    "proj_outproj_144_p4": (
        144,
        144,
        128,
        2,
        (0, 1152, 1440, -1, 126, 14, 2, 144, 2, 1, 288, 288, 4),
    ),
    "proj_ffn1_144_p4": (144, 288, 128, 2, (0, 1152, 1728, -1, 126, 14, 2, 288, 2, 1, 288, 576, 4)),
    "proj_ffn2_144_p4": (288, 144, 128, 2, (0, 2304, 2736, -1, 126, 30, 3, 144, 3, 2, 576, 288, 4)),
    "proj_qkv_192_p4": (192, 576, 64, 1, (0, 768, 1920, -1, 126, 62, 2, 576, 2, 1, 192, 576, 4)),
    "proj_outproj_192_p4": (
        192,
        192,
        64,
        1,
        (0, 768, 1152, -1, 126, 62, 2, 192, 2, 1, 192, 192, 4),
    ),
    "proj_ffn1_192_p4": (192, 384, 64, 1, (0, 768, 1536, -1, 126, 62, 2, 384, 2, 1, 192, 384, 4)),
    "proj_ffn2_192_p4": (384, 192, 64, 1, (0, 1536, 2112, -1, 126, 126, 3, 192, 3, 2, 384, 192, 4)),
    "proj_qkv_240_p4": (240, 720, 16, 1, (0, 960, 2400, -1, 126, 110, 2, 720, 2, 1, 240, 720, 4)),
    "proj_outproj_240_p4": (
        240,
        240,
        16,
        1,
        (0, 960, 1440, -1, 126, 110, 2, 240, 2, 1, 240, 240, 4),
    ),
    "proj_ffn1_240_p4": (240, 480, 16, 1, (0, 960, 1920, -1, 126, 110, 2, 480, 2, 1, 240, 480, 4)),
    "proj_ffn2_240_p4": (480, 240, 16, 1, (0, 1920, 2880, -1, 126, 94, 4, 240, 4, 3, 480, 240, 4)),
}


@pytest.mark.parametrize("name", sorted(PROJECTIONS))
def test_the_projections_abi_reproduces_todays_crs(tmp_path, name):
    k, n_out, n_tok, n_tg, crs = PROJECTIONS[name]
    toml = PROJECTION.format(k=k, n_out=n_out, n_tg=n_tg, n_tok=n_tok)
    m = kernel(tmp_path, toml, family="projections", name=name)
    # As E07-02 runs it: logical shapes, the shared transforms, packing's row counts.
    plan = plan_case(m, {}, FP32)
    placement = place(plan, {n: stored_rows(t, plan) for n, t in m.tensors.items()})
    got = {int(w.register[2:]): w.value for w in register_writes(plan, placement)}
    today = {cr: value & 0xFFFFFFFF for cr, value in enumerate(crs, start=2)}
    # The harness writes the default DS in CR15.L, the state's reset value.
    assert got == {**today, 15: IpuState().regfile.get_cr(15)}
